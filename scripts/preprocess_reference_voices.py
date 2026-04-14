#!/usr/bin/env python3
"""
Preprocess reference voice files for cloning.

Cleans each audio file by:
- DC removal
- High-pass filter (55 Hz cutoff)
- Optional spectral denoising
- Loudness normalization (-20 dBFS target)

Optionally runs through DeepFilterNet for aggressive denoising.

Usage:
  python scripts/preprocess_reference_voices.py
  python scripts/preprocess_reference_voices.py --denoise-strength 0.9
  python scripts/preprocess_reference_voices.py --deepfilter-bin C:/path/to/deep-filter.exe --deepfilter-model C:/path/to/DeepFilterNet3.tar.gz

The script processes all .wav and .mp3 files in reference_voices/
and saves cleaned versions back in place (originals moved to reference_voices_backup/).
"""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


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
        print(f"    [noisereduce not installed, skipping spectral denoise]")
        return audio

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


def load_audio(path: Path):
    """Load audio file with fallback support."""
    try:
        audio, sr = sf.read(str(path), dtype=np.float32)
        return audio, sr
    except Exception as e:
        print(f"    [failed to load: {e}]")
        return None, None


def run_deepfilter(binary: Path, model: str | None, input_path: Path, output_path: Path):
    if not binary.exists():
        raise FileNotFoundError(
            f"DeepFilterNet binary not found: {binary}\n"
            "Download from: https://github.com/Rikorose/DeepFilterNet/releases"
        )
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "-o", str(output_dir)]
    if model:
        cmd.extend(["-m", model])
    cmd.append(str(input_path))
    subprocess.run(cmd, check=True, capture_output=True)
    # DeepFilterNet outputs to output_dir/{input_filename}
    # Move it to the target path
    deepfilter_out = output_dir / input_path.name
    if deepfilter_out.exists() and deepfilter_out != output_path:
        shutil.move(str(deepfilter_out), str(output_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voices", type=Path, default=Path("reference_voices"))
    parser.add_argument("--backup", type=Path, default=Path("reference_voices_backup"))
    parser.add_argument("--denoise-strength", type=float, default=0.8)
    parser.add_argument("--target-dbfs", type=float, default=-20.0)
    parser.add_argument("--target-sr", type=int, default=None,
                        help="Resample to this SR (default: keep original)")
    parser.add_argument("--deepfilter-bin", type=Path, default=None)
    parser.add_argument("--deepfilter-model", type=str, default=None)
    parser.add_argument("--skip-backup", action="store_true",
                        help="Don't back up originals (overwrite in place)")
    args = parser.parse_args()

    voices_dir = args.voices
    backup_dir = args.backup

    if not voices_dir.exists():
        print(f"Voice directory not found: {voices_dir}")
        return

    # Find all audio files
    audio_files = []
    for voice_dir in sorted(voices_dir.iterdir()):
        if not voice_dir.is_dir():
            continue
        audio_files.extend(voice_dir.glob("*.wav"))
        audio_files.extend(voice_dir.glob("*.mp3"))

    if not audio_files:
        print(f"No audio files found in {voices_dir}")
        return

    print(f"\nPreprocessing {len(audio_files)} reference voice files")
    print(f"  Denoise strength: {args.denoise_strength}")
    print(f"  Target loudness: {args.target_dbfs} dBFS")
    if args.target_sr:
        print(f"  Target sample rate: {args.target_sr} Hz")
    if args.deepfilter_bin:
        print(f"  DeepFilterNet: {args.deepfilter_bin}")
    print()

    manifest = []
    success_count = 0
    failed_count = 0

    for i, src_path in enumerate(sorted(audio_files), start=1):
        relative = src_path.relative_to(voices_dir)
        print(f"[{i}/{len(audio_files)}] {relative}...", end=" ", flush=True)

        # Load
        audio, sr = load_audio(src_path)
        if audio is None:
            print("FAILED")
            failed_count += 1
            continue

        audio = ensure_mono_float32(audio)
        orig_sr = sr

        # Backup if needed
        if not args.skip_backup:
            backup_path = backup_dir / relative
            if not backup_path.exists():
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, backup_path)

        # Clean
        cleaned = remove_dc(audio)
        cleaned = highpass(cleaned, sr, cutoff_hz=55.0)
        cleaned = maybe_spectral_denoise(cleaned, sr, args.denoise_strength)
        cleaned = normalize_loudness(cleaned, target_dbfs=args.target_dbfs)

        # Optional resample
        if args.target_sr and args.target_sr != sr:
            cleaned = resample_audio(cleaned, sr, args.target_sr)
            sr = args.target_sr

        # Optional DeepFilterNet
        df_status = "not_run"
        if args.deepfilter_bin:
            temp_path = src_path.with_stem(src_path.stem + "_temp_cleaned")
            write_wav(temp_path, cleaned, sr)
            try:
                temp_df = src_path.with_stem(src_path.stem + "_temp_df")
                run_deepfilter(args.deepfilter_bin, args.deepfilter_model, temp_path, temp_df)
                df_audio, df_sr = load_audio(temp_df)
                if df_audio is not None:
                    cleaned = df_audio
                    sr = df_sr
                    df_status = "ok"
                    temp_df.unlink()
                else:
                    df_status = "load_failed"
            except subprocess.CalledProcessError as e:
                df_status = f"failed:{e.returncode}"
            except Exception as e:
                df_status = f"error:{e}"
            finally:
                if temp_path.exists():
                    temp_path.unlink()

        # Write cleaned version back
        try:
            write_wav(src_path, cleaned, sr)
            print(f"OK{' [DF: ' + df_status + ']' if df_status != 'not_run' else ''}")
            success_count += 1
            manifest.append({
                "file": str(relative),
                "original_sr": orig_sr,
                "cleaned_sr": sr,
                "denoise_strength": args.denoise_strength,
                "deepfilter_status": df_status,
            })
        except Exception as e:
            print(f"WRITE FAILED: {e}")
            failed_count += 1

    # Summary
    print(f"\n{'-'*60}")
    print(f"Processed: {success_count} OK, {failed_count} failed")
    if not args.skip_backup:
        print(f"Backups: {backup_dir}")
    print(f"{'-'*60}\n")

    if success_count > 0:
        manifest_path = voices_dir.parent / "preprocess_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
