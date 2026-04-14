"""
Inflect Enhancer — Dataset

EnhancerDataset  : loads (degraded, clean) audio pairs from a manifest CSV
build_manifest   : helper to create the manifest from two directories of files
"""

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

try:
    from scipy.io import wavfile as _sciwav
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Audio I/O ─────────────────────────────────────────────────────────────────

def _load_wav(path: str | Path, target_sr: int) -> torch.Tensor:
    """Load audio → mono float32 torch Tensor [T], resampled to target_sr."""
    path = str(path)
    if _HAS_SF:
        audio, sr = sf.read(path, dtype="float32")
    elif _HAS_SCIPY:
        sr, audio = _sciwav.read(path)
        audio = audio.astype(np.float32)
        if audio.dtype == np.int16:
            audio = audio / 32768.0
        elif audio.dtype == np.int32:
            audio = audio / 2147483648.0
    else:
        raise RuntimeError("Install soundfile or scipy: pip install soundfile")

    # mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    wav = torch.from_numpy(audio)

    # resample if needed (simple polyphase via torch)
    if sr != target_sr:
        wav = _resample(wav, sr, target_sr)

    return wav


def _resample(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return wav
    from scipy.signal import resample_poly
    g = math.gcd(src_sr, dst_sr)
    out = resample_poly(wav.numpy(), dst_sr // g, src_sr // g)
    return torch.from_numpy(out.astype(np.float32))


# ── Augmentation ──────────────────────────────────────────────────────────────

def _augment_as_degraded(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """
    Simulate TTS-style degradation on a clean audio clip.
    Used to expand the dataset without additional Resemble Enhance API calls.
    - Low-pass filter at 8-12 kHz (thin spectrum)
    - Mild compression (reduce dynamic range)
    - Tiny white noise (SNR ~35-40 dB, barely audible)
    """
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError:
        return wav  # skip augmentation if scipy missing

    wav_np = wav.numpy()

    # Random low-pass at 8-12 kHz
    cutoff = random.uniform(8000, 12000)
    sos = butter(4, cutoff / (sr / 2), btype="low", output="sos")
    wav_np = sosfiltfilt(sos, wav_np).astype(np.float32)

    # Mild dynamic compression: soft clip at ±0.5, rescale
    wav_np = np.tanh(wav_np * 2.0) * 0.5

    # Tiny white noise
    noise_amp = np.sqrt(np.mean(wav_np ** 2) + 1e-8) * 10 ** (-35 / 20)
    wav_np = wav_np + np.random.randn(*wav_np.shape).astype(np.float32) * noise_amp

    return torch.from_numpy(wav_np)


# ── Dataset ───────────────────────────────────────────────────────────────────

class EnhancerDataset(Dataset):
    """
    Loads (degraded, clean) audio pairs from a manifest CSV.

    Manifest columns: degraded_path, clean_path, duration_s
    All clips are trimmed/padded to clip_seconds at sample_rate.

    Use augment=True to synthetically degrade clean clips (2-3× more data,
    no additional API calls needed).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int = 48_000,
        clip_seconds: float = 4.0,
        augment: bool = False,
        augment_ratio: float = 0.5,   # fraction of batch from augmented clean clips
        seed: int = 42,
    ):
        self.sr          = sample_rate
        self.clip_len    = int(clip_seconds * sample_rate)
        self.augment     = augment
        self.augment_ratio = augment_ratio

        rows = []
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append((row["degraded_path"], row["clean_path"]))

        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def _trim_pad(self, wav: torch.Tensor) -> torch.Tensor:
        T = self.clip_len
        if wav.shape[0] >= T:
            # random crop
            start = random.randint(0, wav.shape[0] - T)
            return wav[start: start + T]
        # pad with zeros
        return torch.nn.functional.pad(wav, (0, T - wav.shape[0]))

    def __getitem__(self, idx: int) -> dict:
        deg_path, clean_path = self.rows[idx]

        # Use augmented-from-clean as degraded sometimes (expand dataset)
        if self.augment and random.random() < self.augment_ratio:
            clean = _load_wav(clean_path, self.sr)
            clean = self._trim_pad(clean)
            degraded = _augment_as_degraded(clean.clone(), self.sr)
        else:
            degraded = _load_wav(deg_path,   self.sr)
            clean    = _load_wav(clean_path, self.sr)
            degraded = self._trim_pad(degraded)
            clean    = self._trim_pad(clean)

        return {
            "wav_degraded": degraded,  # [T]
            "wav_clean":    clean,     # [T]
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "wav_degraded": torch.stack([b["wav_degraded"] for b in batch]),
        "wav_clean":    torch.stack([b["wav_clean"]    for b in batch]),
    }


# ── Manifest builder ──────────────────────────────────────────────────────────

def build_manifest(
    degraded_dir: str | Path,
    clean_dir:    str | Path,
    out_csv:      str | Path,
    extensions:   tuple = (".wav", ".mp3"),
):
    """
    Match degraded/clean files by filename stem, write manifest CSV.

    Expects:
      degraded_dir/<name>.wav  →  clean_dir/<name>.wav

    If clean_dir is None or a file doesn't have a clean counterpart, it's skipped.
    """
    degraded_dir = Path(degraded_dir)
    clean_dir    = Path(clean_dir)
    out_csv      = Path(out_csv)

    clean_stems = {}
    for ext in extensions:
        for f in clean_dir.glob(f"*{ext}"):
            clean_stems[f.stem] = f

    rows = []
    for ext in extensions:
        for deg_file in sorted(degraded_dir.glob(f"*{ext}")):
            stem = deg_file.stem
            if stem in clean_stems:
                rows.append({
                    "degraded_path": str(deg_file),
                    "clean_path":    str(clean_stems[stem]),
                    "duration_s":    "",
                })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["degraded_path", "clean_path", "duration_s"])
        w.writeheader()
        w.writerows(rows)

    print(f"Manifest: {len(rows)} pairs → {out_csv}")
    return len(rows)
