"""
Build a small CREMA-D restoration benchmark set and run a first-pass cleanup chain.

Why this exists:
- The AbstractTTS/CREMA-D mirror is an auto-converted parquet wrapper on Hugging Face.
- CREMA-D itself is an original 16 kHz / 16-bit mono WAV corpus.
- Before spending time on model training, we want a controlled A/B set that tests:
  1. baseline original clip
  2. cleanup-only
  3. cleanup + optional LavaSR

This script intentionally keeps the first pass simple and reproducible:
- pull a small sample from the HF dataset mirror
- write raw clips to disk
- apply DC removal, peak guard, high-pass, optional spectral denoise, loudness normalization
- optionally hand off raw or cleaned files to an external DeepFilterNet binary
- optionally hand off cleaned files to an external LavaSR command

Example:
  python scripts/restore_cremad_sample.py --count 12

Optional DeepFilterNet handoff:
  python scripts/restore_cremad_sample.py ^
      --count 10 ^
      --deepfilter-bin "C:\\path\\to\\deep-filter.exe" ^
      --deepfilter-model C:\\path\\to\\DeepFilterNet3.tar.gz

Optional LavaSR handoff:
  python scripts/restore_cremad_sample.py ^
      --lavasr-template "python path\\to\\lavasr_infer.py --input {input} --output {output}"
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "outputs" / "cremad_restore"
HF_DATASET = "AbstractTTS/CREMA-D"


def dbfs_to_amp(db: float) -> float:
    return 10 ** (db / 20.0)


def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    gcd = math.gcd(src_sr, dst_sr)
    up = dst_sr // gcd
    down = src_sr // gcd
    return signal.resample_poly(audio, up, down).astype(np.float32)


def remove_dc(audio: np.ndarray) -> np.ndarray:
    return (audio - np.mean(audio)).astype(np.float32)


def highpass(audio: np.ndarray, sr: int, cutoff_hz: float = 55.0) -> np.ndarray:
    sos = signal.butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def normalize_loudness(audio: np.ndarray, target_dbfs: float = -20.0, peak_limit: float = 0.98) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))
    gain = dbfs_to_amp(target_dbfs) / max(rms, 1e-8)
    out = audio * gain
    peak = float(np.max(np.abs(out)) + 1e-12)
    if peak > peak_limit:
        out = out * (peak_limit / peak)
    return out.astype(np.float32)


def maybe_spectral_denoise(audio: np.ndarray, sr: int, strength: float) -> np.ndarray:
    if strength <= 0:
        return audio
    try:
        import noisereduce as nr
    except Exception:
        return audio

    # Use the quietest 250 ms as the noise profile if available.
    win = max(1, int(0.25 * sr))
    if audio.shape[0] <= win:
        noise = audio
    else:
        energies = np.convolve(audio ** 2, np.ones(win, dtype=np.float32), mode="valid")
        idx = int(np.argmin(energies))
        noise = audio[idx : idx + win]

    reduced = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise,
        prop_decrease=float(np.clip(strength, 0.0, 1.0)),
        stationary=True,
        n_fft=1024,
    )
    return np.asarray(reduced, dtype=np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr, format="WAV", subtype="PCM_16")


def load_dataset_rows(count: int, seed: int):
    try:
        from datasets import DownloadConfig, load_dataset
    except Exception as exc:
        raise RuntimeError("Missing dependency: datasets. Install with `pip install datasets`.") from exc

    try:
        ds = load_dataset(
            HF_DATASET,
            split="train",
            download_config=DownloadConfig(local_files_only=True),
        )
    except Exception:
        ds = load_dataset(HF_DATASET, split="train")

    # Sample a balanced-ish set across emotions and genders instead of random duplicates.
    buckets: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(ds):
        key = (row["major_emotion"], row["gender"])
        buckets.setdefault(key, []).append(idx)

    rng = random.Random(seed)
    picks: list[int] = []
    keys = sorted(buckets.keys())
    while len(picks) < count:
        progressed = False
        for key in keys:
            bucket = buckets[key]
            if bucket:
                picks.append(bucket.pop(rng.randrange(len(bucket))))
                progressed = True
                if len(picks) >= count:
                    break
        if not progressed:
            break

    return [ds[i] for i in picks]


def run_lavasr(template: str, input_path: Path, output_path: Path):
    cmd = template.format(input=str(input_path), output=str(output_path))
    subprocess.run(cmd, shell=True, check=True)


def run_deepfilter(binary: Path, model: str | None, input_path: Path, output_dir: Path):
    if not binary.exists():
        raise FileNotFoundError(
            f"DeepFilterNet binary not found: {binary}\n"
            "Download the Windows release from https://github.com/Rikorose/DeepFilterNet/releases "
            "and pass the real extracted deep-filter.exe path."
        )
    cmd = [str(binary), "-o", str(output_dir)]
    if model:
        cmd.extend(["-m", model])
    cmd.append(str(input_path))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sr", type=int, default=48000)
    parser.add_argument("--denoise-strength", type=float, default=0.8)
    parser.add_argument("--target-dbfs", type=float, default=-20.0)
    parser.add_argument("--deepfilter-bin", type=Path, default=None)
    parser.add_argument(
        "--deepfilter-model",
        type=str,
        default=None,
        help="Path to a DeepFilterNet model tar.gz. Omit to use the binary default.",
    )
    parser.add_argument(
        "--deepfilter-input",
        choices=["raw", "clean"],
        default="raw",
        help="Which stage to feed into DeepFilterNet.",
    )
    parser.add_argument(
        "--lavasr-template",
        type=str,
        default=None,
        help="External command template with {input} and {output} placeholders.",
    )
    args = parser.parse_args()

    run_dir = OUT_ROOT / f"sample_{args.count:02d}_seed_{args.seed}"
    raw_dir = run_dir / "raw"
    raw48_dir = run_dir / "raw_48k"
    clean_dir = run_dir / "clean"
    df_dir = run_dir / "deepfilter"
    sr_dir = run_dir / "lavasr"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = load_dataset_rows(args.count, args.seed)
    manifest = []

    for i, row in enumerate(rows, start=1):
        audio_data = row["audio"]["array"]
        src_sr = int(row["audio"]["sampling_rate"])
        audio = ensure_mono_float32(audio_data)

        base_name = row["file"].replace(".wav", "")
        raw_path = raw_dir / f"{i:02d}_{base_name}.wav"
        clean_path = clean_dir / f"{i:02d}_{base_name}.wav"
        sr_path = sr_dir / f"{i:02d}_{base_name}.wav"

        # Raw save in source rate for honest A/B reference.
        write_wav(raw_path, audio, src_sr)
        raw_48k = resample_audio(audio, src_sr, args.target_sr)
        raw48_path = raw48_dir / f"{i:02d}_{base_name}.wav"
        write_wav(raw48_path, raw_48k, args.target_sr)

        cleaned = remove_dc(audio)
        cleaned = highpass(cleaned, src_sr, cutoff_hz=55.0)
        cleaned = maybe_spectral_denoise(cleaned, src_sr, args.denoise_strength)
        cleaned = normalize_loudness(cleaned, target_dbfs=args.target_dbfs)
        cleaned = resample_audio(cleaned, src_sr, args.target_sr)
        write_wav(clean_path, cleaned, args.target_sr)

        df_status = "not_run"
        df_out_path = df_dir / (raw48_path.name if args.deepfilter_input == "raw" else clean_path.name)
        if args.deepfilter_bin:
            try:
                df_input = raw48_path if args.deepfilter_input == "raw" else clean_path
                run_deepfilter(args.deepfilter_bin, args.deepfilter_model, df_input, df_dir)
                df_status = "ok"
            except subprocess.CalledProcessError as exc:
                df_status = f"failed:{exc.returncode}"

        lavasr_status = "not_run"
        if args.lavasr_template:
            try:
                lavasr_in = df_out_path if df_status == "ok" else clean_path
                run_lavasr(args.lavasr_template, lavasr_in, sr_path)
                lavasr_status = "ok"
            except subprocess.CalledProcessError as exc:
                lavasr_status = f"failed:{exc.returncode}"

        manifest.append(
            {
                "index": i,
                "file": row["file"],
                "major_emotion": row["major_emotion"],
                "gender": row["gender"],
                "transcription": row["transcription"],
                "source_sr": src_sr,
                "raw_path": str(raw_path),
                "raw_48k_path": str(raw48_path),
                "clean_path": str(clean_path),
                "deepfilter_path": str(df_out_path) if args.deepfilter_bin else None,
                "deepfilter_status": df_status,
                "lavasr_path": str(sr_path) if args.lavasr_template else None,
                "lavasr_status": lavasr_status,
            }
        )

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(manifest)} samples to {run_dir}")
    print(f"Manifest: {manifest_path}")
    print("Listen to raw/ vs clean/ first.")
    if args.deepfilter_bin:
        print("Then compare raw/ or clean/ against deepfilter/.")
    if args.lavasr_template:
        print("Then compare deepfilter/ or clean/ against lavasr/.")


if __name__ == "__main__":
    main()
