"""
Inflect Phase 2 v2 — Emotion Delta Vectors (Kokoro Style Space)

FIXES the v1 bug: deltas are now computed AFTER the adapter,
in Kokoro's actual style space. This means adding them to the
style tensor actually works.

Pipeline per clip:
    wav -> mel -> encoder.embed() -> adapter.forward_embed() -> [256] in Kokoro space
                                                                         ^
                                                            delta computed HERE (not before adapter)

Runtime: ~5-10 min on RTX 3060.
Output: emotion-deltas/deltas_v2.pt
"""

import sys
from pathlib import Path

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
VOICE_DIR    = ENCODER_DIR / "data" / "kokoro_voices"

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
        log_mel = torch.log(mel + 1e-5).squeeze(0)
        T_ = log_mel.shape[1]
        if T_ >= 160:
            log_mel = log_mel[:, :160]
        else:
            log_mel = F.pad(log_mel, (0, 160 - T_))
        return log_mel
    except Exception:
        return None


def encode_and_adapt(paths: list[str], encoder, adapter, target_norm: float,
                     batch_size: int = 32) -> torch.Tensor:
    """
    wav paths -> [N, 256] tensors in Kokoro style space (scaled by target_norm).
    This is the space where emotion deltas actually work.
    """
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
        batch = torch.stack(mels).to(DEVICE)
        with torch.no_grad():
            enc_emb = encoder.embed(batch)             # [B, 256] encoder space
            adapted = adapter.forward_embed(enc_emb)   # [B, 256] Kokoro style space (L2-norm)
            adapted = adapted * target_norm            # scale to match real voice packs
        all_embeds.append(adapted.cpu())

    if not all_embeds:
        return torch.zeros(0, 256)
    return torch.cat(all_embeds, dim=0)


def main():
    # ── Load encoder + adapter ────────────────────────────────────────────────
    from train_ve import VoiceEncoder
    from train_1b_adapter_v2 import StyleAdapterV2

    ckpt_aug = ENCODER_DIR / "checkpoints" / "1c_aug_best.pt"
    ckpt_1c  = ENCODER_DIR / "checkpoints" / "1c_best.pt"

    if ckpt_aug.exists():
        ckpt = torch.load(ckpt_aug, map_location=DEVICE, weights_only=False)
        label = f"1c_aug (epoch {ckpt.get('epoch','?')}, val_sim={ckpt.get('val_sim',0):.4f})"
    elif ckpt_1c.exists():
        ckpt = torch.load(ckpt_1c, map_location=DEVICE, weights_only=False)
        label = f"1c (epoch {ckpt.get('epoch','?')}, val_sim={ckpt.get('val_sim',0):.4f})"
    else:
        raise FileNotFoundError("No 1c checkpoint found.")

    encoder = VoiceEncoder().to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    adapter = StyleAdapterV2().to(DEVICE)
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()

    print(f"Loaded: {label}")

    # ── Compute target norm from real voice packs ─────────────────────────────
    pack_norms = []
    for vf in sorted(VOICE_DIR.glob("**/*.pt")):
        pack = torch.load(vf, map_location="cpu", weights_only=True)
        pack_norms.append(pack[:, 0, :].norm(dim=-1).mean().item())
    target_norm = sum(pack_norms) / len(pack_norms)
    print(f"Voice pack mean norm: {target_norm:.4f}")

    # ── Load metadata ─────────────────────────────────────────────────────────
    df = pd.read_csv(METADATA)
    print(f"Metadata: {len(df)} clips, emotions: {df['emotion'].unique()}")

    def resolve_path(p: str) -> str:
        p = Path(p)
        if p.exists():
            return str(p)
        candidate = EXPRESSO_WAV / p.name
        if candidate.exists():
            return str(candidate)
        return str(EXPRESSO_WAV / Path(p).name)

    df["resolved_path"] = df["audio_path"].apply(resolve_path)
    emotions = df["emotion"].unique().tolist()

    # ── Neutral baseline ──────────────────────────────────────────────────────
    neutral_candidates = [e for e in emotions if any(k in e.lower() for k in ["default", "read", "neutral"])]
    if neutral_candidates:
        neutral_emotion = neutral_candidates[0]
        print(f"Neutral baseline: '{neutral_emotion}'")
        neutral_paths = df[df["emotion"] == neutral_emotion]["resolved_path"].tolist()
    else:
        print("No explicit neutral. Using global mean as baseline.")
        neutral_paths = df["resolved_path"].tolist()

    print(f"Encoding neutral baseline ({len(neutral_paths)} clips) in Kokoro style space...")
    neutral_embeds = encode_and_adapt(neutral_paths, encoder, adapter, target_norm)
    if len(neutral_embeds) == 0:
        print("ERROR: Could not encode neutral clips.")
        return

    neutral_mean = neutral_embeds.mean(dim=0)  # [256] — NOT normalized, preserves scale
    print(f"Neutral mean norm: {neutral_mean.norm():.4f}")

    # ── Compute deltas in Kokoro style space ──────────────────────────────────
    deltas = {}
    stats  = {}

    for emotion in tqdm(emotions, desc="Computing deltas"):
        clips = df[df["emotion"] == emotion]["resolved_path"].tolist()
        if len(clips) < 5:
            print(f"  Skip {emotion}: only {len(clips)} clips")
            continue

        embeds = encode_and_adapt(clips, encoder, adapter, target_norm)
        if len(embeds) == 0:
            continue

        emotion_mean = embeds.mean(dim=0)       # [256]
        delta = emotion_mean - neutral_mean     # [256] — in Kokoro style space

        deltas[emotion] = delta
        stats[emotion]  = {
            "n_clips":    len(embeds),
            "delta_norm": delta.norm().item(),
            "cos_sim_to_neutral": F.cosine_similarity(
                F.normalize(emotion_mean.unsqueeze(0), dim=-1),
                F.normalize(neutral_mean.unsqueeze(0), dim=-1),
            ).item(),
        }

    # ── Print stats ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EMOTION DELTA STATS (Kokoro style space)")
    print("=" * 60)
    for e, s in stats.items():
        print(f"  {e:25s}  clips={s['n_clips']:4d}  "
              f"delta_norm={s['delta_norm']:.4f}  "
              f"cos_sim={s['cos_sim_to_neutral']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "deltas":       deltas,
        "neutral_mean": neutral_mean,
        "target_norm":  target_norm,
        "stats":        stats,
        "version":      "v2",
        "note":         "Deltas computed in Kokoro style space (after adapter). "
                        "Add directly to style tensor: style + delta * strength",
    }
    out_path = OUT_DIR / "deltas_v2.pt"
    torch.save(out, out_path)
    print(f"\nSaved: {out_path}")
    print("Run test_cloning.py — it will auto-use deltas_v2.pt")


if __name__ == "__main__":
    main()
