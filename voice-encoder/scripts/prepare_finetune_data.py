"""
Phase 3 data preparation — paralinguistic fine-tuning.

Reads manifest.csv, phonemizes each tagged text (inserting special tag tokens),
loads the target audio as mel spectrogram, and saves everything as a .pt file
ready for train_phase3_paralinguistic.py.

Special tag tokens are appended to Kokoro's vocab starting at ID 200,
leaving a gap above the max existing ID (177) for safety.

Usage:
    python prepare_finetune_data.py
    python prepare_finetune_data.py --manifest ../data/paralinguistic/manifest.csv
    python prepare_finetune_data.py --voices amy arabella   # subset of voices
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

BASE        = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE / "data" / "paralinguistic"
MANIFEST    = DATA_DIR / "manifest.csv"
OUT_FILE    = DATA_DIR / "finetune_dataset.pt"
VOICE_DIR   = BASE / "data" / "kokoro_voices"

TARGET_SR   = 24_000
N_MELS      = 80
HOP_LENGTH  = 256
WIN_LENGTH  = 1024
N_FFT       = 1024

# Tag token IDs — start at 200 to leave a gap above Kokoro's max (177)
# These get added to Kokoro's vocab dict at train time
TAG_TOKENS = {
    "laughs":          200,
    "laughing":        200,
    "laugh":           200,
    "laughs harder":   201,
    "chuckles":        202,
    "giggles":         203,
    "snorts":          204,
    "wheezing":        205,
    "starts laughing": 206,
    "sighs":           207,
    "sigh":            207,
    "exhales":         208,
    "exhale":          208,
    "exhales sharply": 209,
    "frustrated sigh": 210,
    "inhales deeply":  211,
    "whispers":        212,
    "whisper":         212,
    "crying":          213,
    "cry":             213,
    "gasps":           214,
    "gasp":            214,
    "happy gasp":      215,
    "clears throat":   216,
    "muttering":       217,
    "excited":         218,
    "sad":             219,
    "angry":           220,
    "curious":         221,
    "thoughtful":      222,
    "surprised":       223,
    "sarcastic":       224,
    "impressed":       225,
    "warmly":          226,
    "dramatically":    227,
    "delighted":       228,
}

UNIQUE_TAG_IDS = sorted(set(TAG_TOKENS.values()))


def get_mel_transform():
    return T.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        f_min=0.0,
        f_max=8000.0,
    )


def load_audio_mel(path: str, mel_tf) -> torch.Tensor:
    """Load wav, return log-mel [N_MELS, T]."""
    arr, sr = sf.read(path, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    wav = torch.from_numpy(arr).unsqueeze(0)
    if sr != TARGET_SR:
        wav = T.Resample(sr, TARGET_SR)(wav)
    mel = mel_tf(wav)
    return torch.log(mel.squeeze(0) + 1e-5)   # [N_MELS, T]


_VOCAB_CACHE = None

def phonemize_tagged_text(text: str, pipeline, vocab: dict) -> list[int]:
    """
    Convert tagged text to a list of Kokoro vocab token IDs,
    with tag tokens inserted at the right positions.

    Returns list of int token IDs (including new tag token IDs).
    """

    # Split on [tag] markers
    pattern = r'\[([^\]]+)\]'
    parts   = re.split(pattern, text)

    token_ids = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Speech segment — phonemize
            part = part.strip()
            if not part:
                continue
            # Use pipeline to get phoneme string
            try:
                results = list(pipeline(part, voice='af_heart'))
                for r in results:
                    ps = r.phonemes
                    if ps:
                        ids = [vocab[p] for p in ps if p in vocab]
                        token_ids.extend(ids)
            except Exception:
                pass
        else:
            # Tag — insert special token
            tag = part.lower().strip()
            tid = TAG_TOKENS.get(tag)
            if tid is not None:
                token_ids.append(tid)
            else:
                print(f"  [WARN] unknown tag: [{tag}] — skipping")

    return token_ids


def load_voice_style(voice_name: str) -> torch.Tensor | None:
    """Load Kokoro voice pack for this voice, return [1, 256] style vector."""
    # Try local voice dir first
    for pattern in [f"{voice_name}.pt", f"*{voice_name}*.pt"]:
        matches = list(VOICE_DIR.glob(f"**/{pattern}"))
        if matches:
            pack = torch.load(matches[0], map_location="cpu", weights_only=True)
            # pack is [511, 1, 256] — take middle slice
            mid = pack.shape[0] // 2
            return pack[mid, :, :].squeeze(0)  # [256]

    # Map Poe voice names to Kokoro built-in voices
    POE_TO_KOKORO = {
        "amy":       "af_heart",
        "arabella":  "af_nicole",
        "charlotte": "bf_isabella",
        "callum":    "am_michael",
        "bradford":  "bm_fable",
        "austin":    "am_onyx",
        "james":     "bm_george",
        "reginald":  "bm_lewis",
        "rachel":    "af_heart",
        "arnold":    "am_adam",
    }
    kokoro_name = POE_TO_KOKORO.get(voice_name.lower())
    if kokoro_name:
        matches = list(VOICE_DIR.glob(f"**/{kokoro_name}.pt"))
        if matches:
            pack = torch.load(matches[0], map_location="cpu", weights_only=True)
            mid  = pack.shape[0] // 2
            return pack[mid, :, :].squeeze(0)  # [256]

    # Last resort: download via Kokoro pipeline
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from kokoro import KPipeline
        p    = KPipeline(lang_code="a")
        pack = p.load_voice(kokoro_name or "af_heart")
        mid  = pack.shape[0] // 2
        return pack[mid, :, :].squeeze(0)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=str(MANIFEST))
    parser.add_argument("--voices",   nargs="+", default=None,
                        help="Only process these voices (default: all)")
    parser.add_argument("--out",      type=str, default=str(OUT_FILE))
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    from kokoro import KPipeline, KModel
    pipeline = KPipeline(lang_code="a")
    vocab    = KModel().vocab          # load once, reuse for all clips
    mel_tf   = get_mel_transform()

    # Read manifest
    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if args.voices and row["voice"] not in args.voices:
                continue
            rows.append(row)

    print(f"Processing {len(rows)} clips...")

    voices_seen = set(r["voice"] for r in rows)
    voice_styles = {}
    for v in voices_seen:
        style = load_voice_style(v)
        if style is not None:
            voice_styles[v] = style
            print(f"  Loaded style for '{v}': {style.shape}")
        else:
            print(f"  [WARN] No style found for '{v}' — will skip those clips")

    dataset = []
    errors  = 0
    for i, row in enumerate(rows):
        voice = row["voice"]
        if voice not in voice_styles:
            continue

        audio_path = row["audio_path"]
        text       = row["text"]
        tags       = row["tags"].split("|") if row["tags"] else []

        if not Path(audio_path).exists():
            print(f"  [SKIP] missing: {audio_path}")
            errors += 1
            continue

        try:
            # Target mel from ElevenLabs/Poe audio
            mel = load_audio_mel(audio_path, mel_tf)  # [N_MELS, T]

            # Phoneme token IDs with tag tokens injected
            token_ids = phonemize_tagged_text(text, pipeline, vocab)
            if not token_ids:
                print(f"  [SKIP] no tokens for: {text[:40]}")
                errors += 1
                continue

            dataset.append({
                "token_ids":  torch.LongTensor(token_ids),
                "target_mel": mel,                          # [80, T]
                "style":      voice_styles[voice],         # [256]
                "voice":      voice,
                "tags":       tags,
                "text":       text,
            })

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(rows)}  ({len(dataset)} ok, {errors} skipped)")

        except Exception as e:
            print(f"  [ERROR] {audio_path}: {e}")
            errors += 1

    print(f"\nDone: {len(dataset)} clips, {errors} errors")
    print(f"Voices: {dict((v, sum(1 for d in dataset if d['voice']==v)) for v in voices_seen)}")
    print(f"Tag distribution:")
    from collections import Counter
    tag_counts = Counter(t for d in dataset for t in d["tags"] if t)
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag:<22} {count}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "dataset":       dataset,
        "tag_tokens":    TAG_TOKENS,
        "unique_tag_ids": UNIQUE_TAG_IDS,
        "n_mels":        N_MELS,
        "target_sr":     TARGET_SR,
        "hop_length":    HOP_LENGTH,
    }, str(out))
    print(f"\nSaved: {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
