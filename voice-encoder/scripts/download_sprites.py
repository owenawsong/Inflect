"""
Inflect — Download Paralinguistic Sprite Audio Clips (VocalSound, MIT)

VocalSound: 21,024 single-person recordings of laughter, sighs, coughs,
throat clearing, sneezes, sniffs. Clean, isolated, individual voices.

Downloads via HuggingFace streaming (danavery/vocalsound).

Output: voice-encoder/data/sprites/<tag>.wav  (24kHz mono)

Usage:
    python download_sprites.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

BASE       = Path(r"C:\Users\Owen\Inflect-New\voice-encoder")
SPRITE_DIR = BASE / "data" / "sprites"
SPRITE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 24_000

# VocalSound class labels → our tag names (lowercase internally)
# Multiple tags share the same source category
LABEL_TO_TAGS = {
    "laughter":        ["laugh"],
    "sigh":            ["sigh"],
    "cough":           ["cough", "throat"],
    "sneeze":          ["sneeze"],
    "sniff":           ["sniff", "breath"],
    "throat_clearing": ["throat"],
}

CLIPS_PER_LABEL = 30   # stream 30 clips per label, keep 5 best (variety)


def resample_mono(arr: np.ndarray, sr: int) -> np.ndarray:
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    wav = torch.from_numpy(arr.copy()).unsqueeze(0)
    if sr != TARGET_SR:
        wav = T.Resample(sr, TARGET_SR)(wav)
    return wav.squeeze(0).numpy().astype(np.float32)


def trim_silence(audio: np.ndarray, threshold: float = 0.008) -> np.ndarray:
    nonsilent = np.where(np.abs(audio) > threshold)[0]
    if len(nonsilent) == 0:
        return audio
    start = max(0, nonsilent[0] - int(0.04 * TARGET_SR))
    end   = min(len(audio), nonsilent[-1] + int(0.1 * TARGET_SR))
    return audio[start:end]


def normalize(audio: np.ndarray, target_rms: float = 0.12) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms > 1e-6:
        audio = audio * (target_rms / rms)
    return np.clip(audio, -1.0, 1.0)


def pick_best(candidates: list[np.ndarray],
              min_dur: float = 0.3, max_dur: float = 3.0) -> np.ndarray | None:
    """Pick clip with best RMS within duration bounds."""
    valid = [c for c in candidates
             if min_dur <= len(c) / TARGET_SR <= max_dur]
    if not valid:
        valid = candidates  # relax bounds if nothing fits
    if not valid:
        return None
    return max(valid, key=lambda c: float(np.sqrt(np.mean(c ** 2))))


def inspect_sample(sample: dict) -> None:
    """Print sample keys so we know the dataset structure."""
    print("  Dataset fields:", list(sample.keys()))
    for k, v in sample.items():
        if isinstance(v, dict):
            print(f"    {k}: dict with keys {list(v.keys())}")
        elif isinstance(v, (int, float, str)):
            print(f"    {k}: {v!r}")
        else:
            print(f"    {k}: {type(v).__name__}")


def get_audio_array(sample: dict) -> tuple[np.ndarray, int] | None:
    """Extract (array, sr) from a sample dict, handling multiple formats."""
    # Standard HF audio feature (after our audio.py patch returns dict)
    for key in ["audio", "sound", "file"]:
        val = sample.get(key)
        if val is None:
            continue
        if isinstance(val, dict) and "array" in val:
            arr = np.array(val["array"], dtype=np.float32)
            sr  = val.get("sampling_rate", 16000)
            return arr, sr
        if isinstance(val, bytes):
            try:
                arr, sr = sf.read(io.BytesIO(val), dtype="float32")
                return arr, sr
            except Exception:
                continue
    return None


def get_label(sample: dict) -> str | None:
    """Extract label string from sample."""
    for key in ["answer", "label", "class", "category", "target", "classname"]:
        val = sample.get(key)
        if isinstance(val, str):
            return val.lower().replace(" ", "_")
        if isinstance(val, int):
            return str(val)
    return None


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print("Loading VocalSound from HuggingFace (streaming)...")
    print("Using danavery/vocalsound — single-person clean recordings\n")

    ds = load_dataset(
        "lmms-lab/vocalsound",
        split="test",
        streaming=True,
    )

    # Inspect first sample to understand structure
    print("Inspecting dataset structure...")
    first = next(iter(ds))
    inspect_sample(first)
    print()

    # Collect candidates per label
    candidates: dict[str, list[np.ndarray]] = {lbl: [] for lbl in LABEL_TO_TAGS}
    remaining  = {lbl: CLIPS_PER_LABEL for lbl in LABEL_TO_TAGS}

    print("Streaming clips...")
    for sample in ds:
        label = get_label(sample)
        if label is None or label not in remaining or remaining[label] <= 0:
            continue

        result = get_audio_array(sample)
        if result is None:
            continue

        arr, sr = result
        audio = resample_mono(arr, sr)
        audio = trim_silence(audio)

        if len(audio) < int(0.2 * TARGET_SR):  # skip very short
            continue

        candidates[label].append(audio)
        remaining[label] -= 1

        done  = sum(CLIPS_PER_LABEL - r for r in remaining.values())
        total = sum(CLIPS_PER_LABEL for _ in remaining)
        print(f"  [{label}] {CLIPS_PER_LABEL - remaining[label]}/{CLIPS_PER_LABEL}  "
              f"(overall: {done}/{total})", end="\r")

        if all(r <= 0 for r in remaining.values()):
            break

    print("\n\nPicking best 5 clips per label for variety and saving...\n")

    saved = set()
    for label, clips in candidates.items():
        if not clips:
            print(f"  [{label}] no clips — skipping")
            continue

        # Pick 5 best by RMS (diverse voices/ages/genders)
        best_5 = sorted(clips, key=lambda c: float(np.sqrt(np.mean(c ** 2))), reverse=True)[:5]
        best_5 = [normalize(c) for c in best_5]

        tags = LABEL_TO_TAGS[label]
        for tag in tags:
            for i, clip in enumerate(best_5):
                out_path = SPRITE_DIR / f"{tag}_{i}.wav"
                sf.write(str(out_path), clip, TARGET_SR)
                dur = len(clip) / TARGET_SR
                print(f"  [{tag}_{i}] <- {label}  {dur:.2f}s  -> {out_path.name}")
                saved.add(tag)

    print(f"\nDone. Sprites saved: {sorted(saved)}")
    print(f"Location: {SPRITE_DIR}")
    print(f"Each tag now has 5 variants (e.g., laugh_0.wav through laugh_4.wav)")
    print(f"\nTest: python inflect_tts.py \"Hello! [laugh] That is funny.\"")


if __name__ == "__main__":
    main()
