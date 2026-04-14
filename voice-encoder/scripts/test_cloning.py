"""
Inflect - Voice Cloning Test

Usage:
    # Random dataset clip, no emotion
    python test_cloning.py --use-dataset

    # With emotion
    python test_cloning.py --use-dataset --emotion happy
    python test_cloning.py --use-dataset --emotion sad --emotion-strength 1.5

    # Custom audio file
    python test_cloning.py --input path/to/clip.wav --text "Hello world"
    python test_cloning.py --input path/to/clip.wav --emotion whisper

Emotions: happy | sad | whisper

Outputs (saved to outputs/):
    cloned_<speaker>_<emotion>.wav   — zero-shot cloned voice
    reference_<speaker>.wav          — nearest real Kokoro voice (for comparison)
"""

import argparse
import random
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE         = Path(r"C:\Users\Owen\Inflect-New\voice-encoder")
CKPT_DIR     = BASE / "checkpoints"
VOICE_DIR    = BASE / "data" / "kokoro_voices"
_DELTAS_V2   = Path(r"C:\Users\Owen\Inflect-New\emotion-deltas\deltas_v2.pt")
_DELTAS_V1   = Path(r"C:\Users\Owen\Inflect-New\emotion-deltas\deltas.pt")
DELTAS_PATH  = _DELTAS_V2 if _DELTAS_V2.exists() else _DELTAS_V1
OUT_DIR      = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 24_000
N_MELS    = 80

DEFAULT_TEXT = (
    "Hello! My name is being cloned right now. "
    "This is Inflect's zero-shot voice cloning system."
)


# ── Mel helpers ───────────────────────────────────────────────────────────────

_mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR, n_fft=1024, hop_length=256,
    win_length=1024, n_mels=N_MELS, f_max=8000.0,
)

def audio_to_mel(path: str) -> torch.Tensor:
    arr, sr = sf.read(path, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    wav = torch.from_numpy(arr).unsqueeze(0)
    if sr != TARGET_SR:
        wav = T.Resample(sr, TARGET_SR)(wav)
    wav = wav[:, :TARGET_SR * 10]
    mel = _mel_transform(wav)
    return torch.log(mel + 1e-5).squeeze(0)  # [80, T]

def mel_to_batch(mel: torch.Tensor, chunk_frames: int = 160) -> torch.Tensor:
    T_ = mel.shape[1]
    if T_ >= chunk_frames:
        mel = mel[:, :chunk_frames]
    else:
        mel = F.pad(mel, (0, chunk_frames - T_))
    return mel.unsqueeze(0).to(DEVICE)  # [1, 80, 160]


# ── Load models ───────────────────────────────────────────────────────────────

def load_models():
    from train_ve import VoiceEncoder
    from train_1b_adapter_v2 import StyleAdapterV2

    # Priority: 1c_aug_best > 1c_best > adapter_v2_best
    ckpt_aug = CKPT_DIR / "1c_aug_best.pt"
    ckpt_1c  = CKPT_DIR / "1c_best.pt"

    if ckpt_aug.exists():
        ckpt = torch.load(ckpt_aug, map_location=DEVICE, weights_only=False)
        label = f"1c_aug (epoch {ckpt.get('epoch','?')}, val_sim={ckpt.get('val_sim',0):.4f})"
    elif ckpt_1c.exists():
        ckpt = torch.load(ckpt_1c, map_location=DEVICE, weights_only=False)
        label = f"1c (epoch {ckpt.get('epoch','?')}, val_sim={ckpt.get('val_sim',0):.4f})"
    else:
        raise FileNotFoundError("No 1c checkpoint found. Run train_1c_alignment.py first.")

    encoder = VoiceEncoder().to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    adapter = StyleAdapterV2().to(DEVICE)
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()

    print(f"Checkpoint: {label}")
    return encoder, adapter


# ── Voice pack helpers ────────────────────────────────────────────────────────

def load_voice_packs():
    packs = {}
    for vf in sorted(VOICE_DIR.glob("**/*.pt")):
        packs[vf.stem] = torch.load(vf, map_location="cpu", weights_only=True)  # [511,1,256]
    return packs

def target_norm(packs: dict) -> float:
    norms = [p[:, 0, :].norm(dim=-1).mean().item() for p in packs.values()]
    return sum(norms) / len(norms)

def nearest_voice(adapted_emb: torch.Tensor, packs: dict) -> tuple[str, torch.Tensor]:
    """adapted_emb: [256] — find nearest voice pack by centroid cosine sim"""
    adapted_norm = F.normalize(adapted_emb.unsqueeze(0), dim=-1)  # [1,256]
    best_name, best_pack, best_sim = None, None, -1.0
    for name, pack in packs.items():
        centroid = F.normalize(pack[:, 0, :].mean(dim=0, keepdim=True), dim=-1)  # [1,256]
        sim = (adapted_norm * centroid).sum().item()
        if sim > best_sim:
            best_sim, best_name, best_pack = sim, name, pack
    print(f"Nearest voice: {best_name} (cos_sim={best_sim:.4f})")
    return best_name, best_pack


# ── Emotion delta ─────────────────────────────────────────────────────────────

def apply_emotion(style_cpu: torch.Tensor, emotion: str, strength: float) -> torch.Tensor:
    """style_cpu: [511,1,256] — add emotion delta, returns same shape"""
    if not emotion or emotion == "none":
        return style_cpu
    if not DELTAS_PATH.exists():
        print(f"Warning: deltas.pt not found at {DELTAS_PATH}, skipping emotion")
        return style_cpu

    raw = torch.load(DELTAS_PATH, map_location="cpu", weights_only=True)
    deltas = raw["deltas"] if "deltas" in raw else raw
    if emotion not in deltas:
        print(f"Warning: emotion '{emotion}' not in deltas (available: {list(deltas.keys())})")
        return style_cpu

    delta = deltas[emotion]  # [256]
    delta = delta.unsqueeze(0).unsqueeze(0)  # [1,1,256]
    style_cpu = style_cpu + delta * strength
    print(f"Applied emotion: {emotion} (strength={strength:.1f})")
    return style_cpu


# ── Generate audio ────────────────────────────────────────────────────────────

def generate(pipeline, voice_key: str, pack: torch.Tensor, text: str, out_path: Path):
    pipeline.voices[voice_key] = pack
    chunks = []
    for chunk in pipeline(text, voice=voice_key, speed=1.0):
        if chunk.audio is not None:
            chunks.append(chunk.audio)
    if chunks:
        audio = np.concatenate(chunks)
        sf.write(str(out_path), audio, TARGET_SR)
        print(f"  Saved: {out_path}  ({len(audio)/TARGET_SR:.1f}s)")
        return True
    print(f"  No audio generated for {out_path.name}")
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",            type=str,   default=None)
    parser.add_argument("--text",             type=str,   default=DEFAULT_TEXT)
    parser.add_argument("--use-dataset",      action="store_true")
    parser.add_argument("--out",              type=str,   default="cloned")
    parser.add_argument("--emotion",          type=str,   default="none",
                        help="none | happy | sad | whisper")
    parser.add_argument("--emotion-strength", type=float, default=1.0,
                        help="Emotion intensity multiplier (default 1.0, try 0.5-2.0)")
    args = parser.parse_args()

    encoder, adapter = load_models()
    packs     = load_voice_packs()
    norm      = target_norm(packs)
    print(f"Voice pack mean norm: {norm:.4f}")

    from kokoro import KPipeline
    pipeline = KPipeline(lang_code="a")

    # ── Get mel ───────────────────────────────────────────────────────────────
    speaker_id = args.out

    if args.use_dataset:
        df  = pd.read_csv(BASE / "data" / "manifest.csv")
        df  = df[df["duration"] >= 4.0]
        row = df.sample(1).iloc[0]
        speaker_id = str(row["speaker_id"])
        print(f"\nSpeaker: {speaker_id}")
        mel   = torch.load(row["mel_path"], weights_only=True)
        batch = mel_to_batch(mel)
    elif args.input:
        print(f"\nInput: {args.input}")
        mel   = audio_to_mel(args.input)
        batch = mel_to_batch(mel)
    else:
        print("No input given — generating Kokoro baseline with af_heart for reference")
        chunks = []
        for chunk in pipeline(args.text, voice="af_heart"):
            if chunk.audio is not None:
                chunks.append(chunk.audio)
        if chunks:
            out = OUT_DIR / "kokoro_baseline.wav"
            sf.write(str(out), np.concatenate(chunks), TARGET_SR)
            print(f"Saved baseline: {out}")
        sys.exit(0)

    # ── Encode → adapt → rescale ──────────────────────────────────────────────
    with torch.no_grad():
        enc_emb = encoder.embed(batch)           # [1, 256]
        style   = adapter(enc_emb) * norm        # [1, 511, 1, 256]

    style_cpu = style.squeeze(0).cpu()           # [511, 1, 256]

    # ── Find nearest real voice (for reference) ───────────────────────────────
    ref_name, ref_pack = nearest_voice(enc_emb.squeeze(0).cpu(), packs)

    # ── Apply emotion delta ───────────────────────────────────────────────────
    emotion_tag  = args.emotion.lower()
    style_emoted = apply_emotion(style_cpu, emotion_tag, args.emotion_strength)

    # ── Generate files ────────────────────────────────────────────────────────
    suffix = f"_{emotion_tag}" if emotion_tag != "none" else ""
    text   = args.text
    print(f"\nText: {text[:80]}")
    print("Generating...")

    # 1. Cloned voice (with optional emotion)
    clone_path = OUT_DIR / f"cloned_{speaker_id}{suffix}.wav"
    generate(pipeline, "_inflect_clone", style_emoted, text, clone_path)

    # 2. Reference — nearest real Kokoro voice (no emotion, for comparison)
    ref_path = OUT_DIR / f"reference_{speaker_id}_{ref_name}.wav"
    if not ref_path.exists():
        generate(pipeline, "_inflect_ref", ref_pack[:, :, :], text, ref_path)
    else:
        print(f"  Reference exists: {ref_path.name}")

    print(f"\nListen and compare:")
    print(f"  Cloned:    {clone_path.name}")
    print(f"  Reference: {ref_path.name}  (nearest Kokoro voice = {ref_name})")
