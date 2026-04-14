"""
Inflect Phase 2 - Emotion Delta Vectors

Computes per-emotion offset vectors in the 256-dim style space.
Uses Expresso dataset (already downloaded at C:\inflect-data\expresso_24k\).

Method:
    emotion_delta[e] = mean(encode(emotional_clips)) - mean(encode(neutral_clips))

Result: a dict of {emotion: [256-dim tensor]} saved to:
    Inflect-New/emotion-deltas/deltas.pt

Runtime: ~20 min on RTX 3060.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE        = Path(r"C:\Users\Owen\Inflect-New")
ENCODER_DIR = BASE / "voice-encoder"
OUT_DIR     = BASE / "emotion-deltas"
OUT_DIR.mkdir(exist_ok=True)

EXPRESSO_WAV = Path(r"C:\inflect-data\expresso_24k")
METADATA     = Path(r"C:\inflect-data\metadata\train.csv")

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 24_000
N_MELS    = 80

print(f"Device: {DEVICE}")

_mel_t = T.MelSpectrogram(
    sample_rate=TARGET_SR, n_fft=1024, hop_length=256,
    win_length=1024, n_mels=N_MELS, f_max=8000.0,
)

def wav_to_mel(path: str) -> torch.Tensor | None:
    try:
        arr, sr = sf.read(path, dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        wav = torch.from_numpy(arr).unsqueeze(0)
        if sr != TARGET_SR:
            wav = T.Resample(sr, TARGET_SR)(wav)
        wav = wav[:, :TARGET_SR * 10]
        mel = _mel_t(wav)
        log_mel = torch.log(mel + 1e-5).squeeze(0)  # [80, T]
        # Crop/pad to 160 frames
        T_ = log_mel.shape[1]
        if T_ >= 160:
            log_mel = log_mel[:, :160]
        else:
            log_mel = F.pad(log_mel, (0, 160 - T_))
        return log_mel
    except Exception:
        return None


def encode_clips(paths: list[str], encoder, batch_size: int = 32) -> torch.Tensor:
    """Encode a list of audio paths -> stacked [N, 256] embeddings."""
    all_embeds = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        mels = []
        for p in batch_paths:
            mel = wav_to_mel(p)
            if mel is not None:
                mels.append(mel)
        if not mels:
            continue
        batch = torch.stack(mels).to(DEVICE)  # [B, 80, 160]
        with torch.no_grad():
            embeds = encoder.embed(batch)  # [B, 256]
        all_embeds.append(embeds.cpu())

    if not all_embeds:
        return torch.zeros(0, 256)
    return torch.cat(all_embeds, dim=0)


def main():
    # Load encoder
    from train_ve import VoiceEncoder
    encoder = VoiceEncoder().to(DEVICE)
    ckpt = torch.load(
        ENCODER_DIR / "checkpoints" / "ve_latest.pt",
        map_location=DEVICE, weights_only=False
    )
    encoder.load_state_dict(ckpt["model"])
    encoder.eval()
    print(f"Encoder loaded (epoch {ckpt.get('epoch','?')})")

    # Load metadata
    df = pd.read_csv(METADATA)
    print(f"Metadata: {len(df)} clips, emotions: {df['emotion'].unique()}")

    # Map audio paths — Expresso audio_path column is relative
    def resolve_path(p: str) -> str:
        p = Path(p)
        # Try direct, then under expresso_24k
        if p.exists():
            return str(p)
        candidate = EXPRESSO_WAV / p.name
        if candidate.exists():
            return str(candidate)
        # Try just filename
        candidate2 = EXPRESSO_WAV / Path(p).name
        if candidate2.exists():
            return str(candidate2)
        return str(p)  # will fail gracefully in wav_to_mel

    df["resolved_path"] = df["audio_path"].apply(resolve_path)

    emotions = df["emotion"].unique().tolist()
    print(f"Emotions found: {emotions}")

    # Find neutral/default emotion (use whichever has the most clips, or "default"/"read")
    # Expresso emotions: happy, sad, whisper — no explicit neutral
    # Use "default" if present, else compute global mean as neutral baseline
    neutral_candidates = [e for e in emotions if "default" in e.lower() or "read" in e.lower() or "neutral" in e.lower()]
    if neutral_candidates:
        neutral_emotion = neutral_candidates[0]
        print(f"Neutral baseline: '{neutral_emotion}'")
        neutral_paths = df[df["emotion"] == neutral_emotion]["resolved_path"].tolist()
    else:
        # No explicit neutral — use all clips as baseline (global mean)
        print("No explicit neutral emotion found. Using global mean as baseline.")
        neutral_paths = df["resolved_path"].tolist()

    print(f"Encoding neutral baseline ({len(neutral_paths)} clips)...")
    neutral_embeds = encode_clips(neutral_paths, encoder)
    if len(neutral_embeds) == 0:
        print("ERROR: Could not encode any neutral clips. Check paths.")
        return

    neutral_mean = F.normalize(neutral_embeds.mean(dim=0), dim=0)  # [256]
    print(f"Neutral mean: {neutral_mean.shape}, norm={neutral_mean.norm():.3f}")

    # Compute delta for each emotion
    deltas = {}
    stats  = {}

    for emotion in tqdm(emotions, desc="Computing deltas"):
        clips = df[df["emotion"] == emotion]["resolved_path"].tolist()
        if len(clips) < 5:
            print(f"  Skip {emotion}: only {len(clips)} clips")
            continue

        embeds = encode_clips(clips, encoder)
        if len(embeds) == 0:
            continue

        emotion_mean = F.normalize(embeds.mean(dim=0), dim=0)  # [256]
        delta = emotion_mean - neutral_mean                     # [256]

        deltas[emotion] = delta
        stats[emotion]  = {
            "n_clips":   len(embeds),
            "delta_norm": delta.norm().item(),
            "cos_sim_to_neutral": F.cosine_similarity(
                emotion_mean.unsqueeze(0), neutral_mean.unsqueeze(0)
            ).item(),
        }

    # Print stats
    print("\n" + "="*55)
    print("EMOTION DELTA STATS")
    print("="*55)
    for e, s in stats.items():
        print(f"  {e:20s}  clips={s['n_clips']:4d}  "
              f"delta_norm={s['delta_norm']:.4f}  "
              f"cos_sim={s['cos_sim_to_neutral']:.4f}")

    # Save
    out = {
        "deltas":       deltas,          # {emotion: [256]}
        "neutral_mean": neutral_mean,    # [256]
        "stats":        stats,
        "note": (
            "emotion_delta[e] = mean(emotional_embeds) - mean(neutral_embeds). "
            "To apply: conditioned_style = speaker_style + delta * intensity"
        ),
    }
    out_path = OUT_DIR / "deltas.pt"
    torch.save(out, out_path)
    print(f"\nSaved: {out_path}")
    print("Phase 2 complete.")


if __name__ == "__main__":
    main()
