"""
Generate (degraded, clean) audio pairs for Inflect Enhancer training.

Takes VoxCPM2 generated clips as the "degraded" input,
runs Resemble Enhance on each to produce the "clean" target.

Output layout:
  outputs/enhancer_pairs/
    degraded/   <-- symlinks or copies of VoxCPM clips
    clean/      <-- Resemble Enhance outputs
    manifest.csv

Usage:
  pip install resemble-enhance
  python scripts/generate_enhancer_pairs.py
  python scripts/generate_enhancer_pairs.py --limit 500  # test with 500 clips first
  python scripts/generate_enhancer_pairs.py --voxcpm-dir outputs/voxcpm_dataset/20260411_141432/audio
"""

import argparse
import csv
import shutil
import time
from pathlib import Path

import torch

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOXCPM_DIR = PROJECT_ROOT / "outputs/voxcpm_dataset"
DEFAULT_OUT_DIR    = PROJECT_ROOT / "outputs/enhancer_pairs"

MANIFEST_COLS = ["degraded_path", "clean_path", "duration_s"]


def _find_audio_files(root: Path) -> list[Path]:
    """Recursively find all .wav files under root."""
    return sorted(root.rglob("*.wav"))


def _load_wav(path: Path):
    if not _HAS_SF:
        raise RuntimeError("pip install soundfile")
    audio, sr = sf.read(str(path), dtype="float32")
    import numpy as np
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return torch.from_numpy(audio), sr


def _save_wav(path: Path, wav: torch.Tensor, sr: int):
    if not _HAS_SF:
        raise RuntimeError("pip install soundfile")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.cpu().numpy(), sr, format="WAV", subtype="PCM_16")


def _duration_s(wav: torch.Tensor, sr: int) -> float:
    return wav.shape[0] / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxcpm-dir", type=Path, default=DEFAULT_VOXCPM_DIR,
                    help="Directory containing VoxCPM2 audio files (searched recursively)")
    ap.add_argument("--out-dir",    type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit",      type=int,  default=None,
                    help="Process at most N clips (for testing)")
    ap.add_argument("--device",     type=str,  default=None)
    ap.add_argument("--nfe",        type=int,  default=64,
                    help="Resemble Enhance CFG steps (64=highest quality, 1=fastest)")
    ap.add_argument("--solver",     type=str,  default="midpoint")
    ap.add_argument("--resume",     action="store_true",
                    help="Skip files already in manifest")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Resemble Enhance
    try:
        from resemble_enhance.enhancer.inference import denoise, enhance
    except ImportError:
        raise SystemExit(
            "resemble-enhance not installed.\n"
            "pip install resemble-enhance"
        )

    # Find VoxCPM audio files
    audio_files = _find_audio_files(args.voxcpm_dir)
    if not audio_files:
        raise SystemExit(f"No .wav files found under {args.voxcpm_dir}")

    if args.limit:
        audio_files = audio_files[:args.limit]

    print(f"Found {len(audio_files)} audio files")

    # Resume support
    out_dir      = args.out_dir
    degraded_dir = out_dir / "degraded"
    clean_dir    = out_dir / "clean"
    degraded_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    done_stems = set()
    if args.resume and manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_stems.add(Path(row["degraded_path"]).stem)
        print(f"Resuming: {len(done_stems)} already done")

    # Write manifest header if needed
    if not manifest_path.exists():
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_COLS).writeheader()

    ok = 0
    err = 0
    t0 = time.time()

    for i, src in enumerate(audio_files, start=1):
        stem = src.stem
        if stem in done_stems:
            continue

        print(f"[{i}/{len(audio_files)}] {src.name}...", end=" ", flush=True)

        try:
            wav, sr = _load_wav(src)

            # Run Resemble Enhance (enhancer only — no denoiser for clean TTS input)
            wav_enhanced, new_sr = enhance(
                wav,
                sr,
                device=device,
                nfe=args.nfe,
                solver=args.solver,
                lambd=0.9,    # 0=denoise only, 1=enhance only; we want enhance
                tau=0.5,
            )

            # Save degraded copy
            deg_path   = degraded_dir / f"{stem}.wav"
            clean_path = clean_dir    / f"{stem}.wav"
            _save_wav(deg_path,   wav,         sr)
            _save_wav(clean_path, wav_enhanced, new_sr)

            dur = _duration_s(wav, sr)

            with open(manifest_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=MANIFEST_COLS).writerow({
                    "degraded_path": str(deg_path),
                    "clean_path":    str(clean_path),
                    "duration_s":    f"{dur:.2f}",
                })

            ok += 1
            elapsed = time.time() - t0
            rate = ok / (elapsed / 3600)
            print(f"OK  ({ok} done, {rate:.0f}/hr)")

        except Exception as e:
            err += 1
            print(f"FAIL: {e}")

    print(f"\nDone. {ok} pairs saved, {err} errors")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext: python inflect/enhancer/train.py --stage 1")


if __name__ == "__main__":
    main()
