"""
Inflect Voice Encoder - Data Preparation
Downloads LibriSpeech clean-100 + VCTK, resamples to 24kHz,
computes 80-bin log-mel spectrograms, organizes by speaker.

Usage:
    python prepare_ve_data.py [--librispeech-only] [--vctk-only]

Output:
    Inflect-New/voice-encoder/data/mels/        <- .pt files
    Inflect-New/voice-encoder/data/manifest.csv
"""

import argparse
import io
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.transforms as T
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE      = Path(r"C:\Users\Owen\Inflect-New\voice-encoder")
DATA_DIR  = BASE / "data"
MEL_DIR   = DATA_DIR / "mels"
MANIFEST  = DATA_DIR / "manifest.csv"
VCTK_TAR  = DATA_DIR / "raw" / "vctk_tars"

TARGET_SR    = 24_000
N_MELS       = 80
N_FFT        = 1024
HOP_LENGTH   = 256
WIN_LENGTH   = 1024
MIN_DURATION = 2.0    # skip clips shorter than this
MAX_DURATION = 15.0   # trim clips longer than this
MAX_PER_SPK  = 300    # cap clips per speaker

MEL_DIR.mkdir(parents=True, exist_ok=True)
VCTK_TAR.mkdir(parents=True, exist_ok=True)

# ── Mel transform (shared) ─────────────────────────────────────────────────────
_mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS,
    f_min=0.0,
    f_max=8000.0,
)

def audio_to_logmel(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """waveform [1, T] at sr → log-mel [n_mels, T']"""
    if sr != TARGET_SR:
        resampler = T.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    max_samples = int(MAX_DURATION * TARGET_SR)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    mel = _mel_transform(waveform)
    return torch.log(mel + 1e-5).squeeze(0)  # [n_mels, T']

def duration_ok(n_samples: int, sr: int) -> bool:
    dur = n_samples / sr
    return MIN_DURATION <= dur <= MAX_DURATION


# ── LibriSpeech clean-100 ──────────────────────────────────────────────────────
def process_librispeech() -> list[dict]:
    print("\n" + "="*60)
    print("LibriSpeech clean-100  (251 speakers, ~100h)")
    print("="*60)

    from datasets import load_dataset, Audio

    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="train.100",
        streaming=True,
    )
    # decode=False → raw bytes; we decode with soundfile (no torchcodec needed)
    ds = ds.cast_column("audio", Audio(decode=False))

    records = []
    speaker_counts: dict[str, int] = {}
    skipped = 0
    processed = 0

    pbar = tqdm(ds, desc="LibriSpeech")
    for item in pbar:
        try:
            speaker_id = str(item["speaker_id"]).strip()

            if speaker_counts.get(speaker_id, 0) >= MAX_PER_SPK:
                skipped += 1
                continue

            audio_bytes = item["audio"]["bytes"]
            arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)

            if not duration_ok(len(arr), sr):
                skipped += 1
                continue

            waveform = torch.from_numpy(arr).unsqueeze(0)  # [1, T]
            log_mel = audio_to_logmel(waveform, sr)

            out_name = f"ls_{speaker_id}_{processed:07d}.pt"
            out_path = MEL_DIR / out_name
            torch.save(log_mel, out_path)

            records.append({
                "dataset":    "librispeech",
                "speaker_id": f"ls_{speaker_id}",
                "mel_path":   str(out_path),
                "duration":   len(arr) / sr,
                "n_frames":   log_mel.shape[1],
            })
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
            processed += 1

            if processed % 500 == 0:
                pbar.set_postfix(ok=processed, skip=skipped, spk=len(speaker_counts))

        except Exception:
            skipped += 1

    print(f"LibriSpeech: {processed} clips, {len(speaker_counts)} speakers, {skipped} skipped")
    return records


# ── VCTK ──────────────────────────────────────────────────────────────────────
def process_vctk() -> list[dict]:
    print("\n" + "="*60)
    print("VCTK  (109 speakers, ~44h)")
    print("="*60)

    from huggingface_hub import hf_hub_download

    # VCTK on HF is packed as tar shards in badayvedat/VCTK
    # Filenames: audio/train-00.tar ... train-11.tar
    tar_names = [f"audio/train-{i:02d}.tar" for i in range(12)]

    records = []
    speaker_counts: dict[str, int] = {}
    skipped = 0
    processed = 0

    for tar_name in tar_names:
        print(f"  Fetching {tar_name} ...")
        try:
            local_tar = hf_hub_download(
                repo_id="badayvedat/VCTK",
                filename=tar_name,
                repo_type="dataset",
                local_dir=str(VCTK_TAR),
            )
        except Exception as e:
            print(f"  Download failed: {e}")
            continue

        with tarfile.open(local_tar, "r") as tf:
            members = [m for m in tf.getmembers() if m.name.endswith((".wav", ".flac"))]
            for member in tqdm(members, desc=f"  {tar_name}", leave=False):
                try:
                    # VCTK filenames: p225/p225_001.wav or p225_001.wav
                    basename = Path(member.name).stem          # p225_001
                    speaker_id = basename.split("_")[0]        # p225
                    if not speaker_id.startswith("p"):
                        skipped += 1
                        continue
                    if speaker_counts.get(speaker_id, 0) >= MAX_PER_SPK:
                        skipped += 1
                        continue

                    f = tf.extractfile(member)
                    if f is None:
                        skipped += 1
                        continue

                    arr, sr = sf.read(io.BytesIO(f.read()), dtype="float32")
                    if arr.ndim > 1:
                        arr = arr.mean(axis=1)

                    if not duration_ok(len(arr), sr):
                        skipped += 1
                        continue

                    waveform = torch.from_numpy(arr).unsqueeze(0)
                    log_mel = audio_to_logmel(waveform, sr)

                    out_name = f"vctk_{speaker_id}_{processed:07d}.pt"
                    out_path = MEL_DIR / out_name
                    torch.save(log_mel, out_path)

                    records.append({
                        "dataset":    "vctk",
                        "speaker_id": f"vctk_{speaker_id}",
                        "mel_path":   str(out_path),
                        "duration":   len(arr) / sr,
                        "n_frames":   log_mel.shape[1],
                    })
                    speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
                    processed += 1

                except Exception:
                    skipped += 1

        # Optional: delete tar after processing to save disk space
        # os.remove(local_tar)

    print(f"VCTK: {processed} clips, {len(speaker_counts)} speakers, {skipped} skipped")
    return records


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-only", action="store_true")
    parser.add_argument("--vctk-only", action="store_true")
    args = parser.parse_args()

    print("Inflect Voice Encoder - Data Preparation")
    print(f"Mel dir:   {MEL_DIR}")
    print(f"Manifest:  {MANIFEST}")
    print(f"SR: {TARGET_SR}Hz | Mels: {N_MELS} | Duration: {MIN_DURATION}-{MAX_DURATION}s")

    all_records = []

    if not args.vctk_only:
        all_records += process_librispeech()

    if not args.librispeech_only:
        all_records += process_vctk()

    df = pd.DataFrame(all_records)
    df.to_csv(MANIFEST, index=False)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total clips:    {len(df):,}")
    print(f"Total speakers: {df['speaker_id'].nunique():,}")
    if "librispeech" in df["dataset"].values:
        mask = df.dataset == "librispeech"
        print(f"  LibriSpeech:  {df[mask]['speaker_id'].nunique()} spk, {mask.sum():,} clips")
    if "vctk" in df["dataset"].values:
        mask = df.dataset == "vctk"
        print(f"  VCTK:         {df[mask]['speaker_id'].nunique()} spk, {mask.sum():,} clips")
    print(f"Total duration: {df['duration'].sum()/3600:.1f}h")
    print(f"Avg clip:       {df['duration'].mean():.1f}s")
    print(f"Manifest:       {MANIFEST}")
    print("\nDone. Run train_ve.py next.")


if __name__ == "__main__":
    main()
