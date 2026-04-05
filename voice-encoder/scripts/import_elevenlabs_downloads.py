"""
Imports manually downloaded ElevenLabs clips and builds the training manifest.

ElevenLabs names downloaded files after the first ~50 chars of the text,
e.g. "I can't believe you actually said that. [laughs] That's the funni.mp3"
This script fuzzy-matches those filenames back to the training corpus texts.

Folder structure expected:
    data/paralinguistic/downloads/
        rachel/
            I can't believe you actually said that. [laughs] That's t....mp3
            Wait, you tripped over your own feet [laughs] In front of....mp3
            ...
        josh/
            ...
        bella/
            ...

Output:
    data/paralinguistic/
        rachel/   ← renamed + converted to .wav
            0000_rachel.wav + 0000_rachel.txt
            ...
        manifest.csv

Usage:
    # Auto-scan data/paralinguistic/downloads/ for all voice subfolders
    python import_elevenlabs_downloads.py

    # Specific folder + voice name
    python import_elevenlabs_downloads.py --folder "C:/Users/Owen/Downloads/rachel_clips" --voice rachel

    # Preview matches without moving/converting anything
    python import_elevenlabs_downloads.py --dry-run

    # Lower match threshold if some files aren't being matched (default: 0.45)
    python import_elevenlabs_downloads.py --threshold 0.35
"""

import argparse
import csv
import re
import shutil
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

BASE      = Path(__file__).resolve().parent.parent
DL_DIR    = BASE / "data" / "paralinguistic" / "downloads"
OUT_DIR   = BASE / "data" / "paralinguistic"
TARGET_SR = 24_000

sys.path.insert(0, str(BASE / "data"))
from training_texts import TEXTS


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_filename(name: str) -> str:
    """Strip extension and sanitize a filename for comparison."""
    name = Path(name).stem                  # drop extension
    name = re.sub(r'[_\-]+', ' ', name)     # _ and - -> space
    name = re.sub(r'\s+', ' ', name).strip()
    return name.lower()


def clean_text(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r'\s+', ' ', text).strip().lower()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def best_match(filename: str, texts: list[str], threshold: float) -> tuple[int, float] | None:
    """Return (index, score) of the best-matching text, or None if below threshold."""
    cleaned_fn = clean_filename(filename)
    best_i, best_score = -1, 0.0
    for i, text in enumerate(texts):
        # Compare against the first 80 chars of the text (that's what ElevenLabs uses for filename)
        preview = clean_text(text[:80])
        score   = similarity(cleaned_fn, preview)
        if score > best_score:
            best_score = score
            best_i     = i
    if best_score >= threshold:
        return best_i, best_score
    return None


def extract_tags(text: str) -> list[str]:
    return re.findall(r'\[([^\]]+)\]', text)


def strip_tags(text: str) -> str:
    return re.sub(r'\[[^\]]+\]', '', text).strip()


def convert_to_wav(src: Path, dst: Path) -> bool:
    """Convert any audio file to 24kHz mono WAV. Returns True on success."""
    try:
        # Try soundfile first (handles WAV, FLAC, OGG)
        audio, sr = sf.read(str(src), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            wav = torch.from_numpy(audio).unsqueeze(0)
            wav = T.Resample(sr, TARGET_SR)(wav)
            audio = wav.squeeze(0).numpy()
        sf.write(str(dst), audio, TARGET_SR)
        return True
    except Exception:
        pass

    # Fallback: try pydub for MP3
    try:
        from pydub import AudioSegment
        seg   = AudioSegment.from_file(str(src))
        seg   = seg.set_channels(1).set_frame_rate(TARGET_SR)
        arr   = np.array(seg.get_array_of_samples(), dtype=np.float32)
        arr  /= 2 ** (seg.sample_width * 8 - 1)
        sf.write(str(dst), arr, TARGET_SR)
        return True
    except Exception as e:
        print(f"    [ERROR] Could not convert {src.name}: {e}")
        return False


# ── Core import logic ─────────────────────────────────────────────────────────

def import_folder(folder: Path, voice_name: str, texts: list[str],
                  threshold: float, dry_run: bool) -> list[dict]:
    """Match all audio files in folder to texts, convert + rename, return manifest rows."""
    audio_exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
    files      = sorted([f for f in folder.iterdir()
                         if f.suffix.lower() in audio_exts])

    if not files:
        print(f"  [!] No audio files found in {folder}")
        return []

    print(f"\n  {len(files)} files found in {folder.name}/")

    out_voice = OUT_DIR / voice_name.lower()
    if not dry_run:
        out_voice.mkdir(parents=True, exist_ok=True)

    rows       = []
    used_texts = set()

    for f in files:
        match = best_match(f.name, texts, threshold)
        if match is None:
            print(f"  [NO MATCH] {f.name[:60]}  (threshold={threshold})")
            continue

        idx, score = match
        if idx in used_texts:
            print(f"  [DUPLICATE] {f.name[:60]} -> text {idx:03d} already used")
            continue

        text  = texts[idx]
        tags  = extract_tags(text)
        plain = strip_tags(text)

        dst_wav = out_voice / f"{idx:04d}_{voice_name.lower()}.wav"
        dst_txt = out_voice / f"{idx:04d}_{voice_name.lower()}.txt"

        status = "DRY RUN" if dry_run else "OK"
        print(f"  [{status}] {f.name[:55]}")
        print(f"         -> {idx:04d}  score={score:.2f}  tags={tags}")

        if not dry_run:
            ok = convert_to_wav(f, dst_wav)
            if ok:
                dst_txt.write_text(text, encoding="utf-8")
                rows.append({
                    "audio_path": str(dst_wav),
                    "text":       text,
                    "plain_text": plain,
                    "voice":      voice_name.lower(),
                    "tags":       "|".join(tags),
                    "match_score": f"{score:.3f}",
                })
                used_texts.add(idx)
        else:
            used_texts.add(idx)
            rows.append({
                "audio_path": str(dst_wav),
                "text":       text,
                "plain_text": plain,
                "voice":      voice_name.lower(),
                "tags":       "|".join(tags),
                "match_score": f"{score:.3f}",
            })

    # Report any unmatched texts
    matched_idxs = {int(r["audio_path"].split("\\")[-1][:4]) for r in rows}
    unmatched    = [i for i in range(len(texts)) if i not in matched_idxs]
    if unmatched:
        print(f"\n  Unmatched texts ({len(unmatched)}): indices {unmatched[:10]}"
              + (" ..." if len(unmatched) > 10 else ""))
        print(f"  Try --threshold lower if files exist but aren't matching.")

    print(f"\n  Matched: {len(rows)}/{len(files)} files for voice '{voice_name}'")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",    type=str, default=None,
                        help="Folder of downloaded files (overrides auto-scan)")
    parser.add_argument("--voice",     type=str, default=None,
                        help="Voice name for --folder mode")
    parser.add_argument("--dl-dir",    type=str, default=str(DL_DIR),
                        help=f"Root downloads dir to auto-scan (default: {DL_DIR})")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Min fuzzy-match score to accept (0-1, default: 0.45)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show matches without converting or moving files")
    args = parser.parse_args()

    manifest_path = OUT_DIR / "manifest.csv"
    all_rows      = []

    # Load existing manifest to avoid duplicates
    existing_keys = set()
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_keys.add(row["audio_path"])
        print(f"Existing manifest: {len(existing_keys)} entries")

    if args.folder:
        # Single folder mode
        folder     = Path(args.folder)
        voice_name = args.voice or folder.name
        rows       = import_folder(folder, voice_name, TEXTS, args.threshold, args.dry_run)
        all_rows.extend(rows)
    else:
        # Auto-scan downloads dir for voice subfolders
        dl_root = Path(args.dl_dir)
        if not dl_root.exists():
            print(f"Downloads dir not found: {dl_root}")
            print(f"Create it and put voice subfolders inside:")
            print(f"  {dl_root}/rachel/   ← your rachel downloads")
            print(f"  {dl_root}/josh/     ← your josh downloads")
            sys.exit(1)

        voice_dirs = [d for d in dl_root.iterdir() if d.is_dir()]
        if not voice_dirs:
            print(f"No voice subfolders found in {dl_root}")
            sys.exit(1)

        print(f"Found {len(voice_dirs)} voice folder(s): {[d.name for d in voice_dirs]}")

        for vdir in sorted(voice_dirs):
            rows = import_folder(vdir, vdir.name, TEXTS, args.threshold, args.dry_run)
            # Filter already in manifest
            new_rows = [r for r in rows if r["audio_path"] not in existing_keys]
            all_rows.extend(new_rows)

    # Write manifest
    if all_rows and not args.dry_run:
        write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
        with open(manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_path", "text", "plain_text",
                                                    "voice", "tags", "match_score"])
            if write_header:
                writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nManifest updated: {manifest_path}  (+{len(all_rows)} rows)")
    elif args.dry_run:
        print(f"\n[DRY RUN] Would add {len(all_rows)} rows to manifest.")
    else:
        print("\nNothing new to add.")

    # Summary by tag
    if all_rows:
        from collections import Counter
        tag_counts = Counter()
        for row in all_rows:
            for tag in row["tags"].split("|"):
                if tag:
                    tag_counts[tag] += 1
        print("\nTag distribution:")
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
            bar = "#" * count
            print(f"  {tag:<20} {count:3d}  {bar}")


if __name__ == "__main__":
    main()
