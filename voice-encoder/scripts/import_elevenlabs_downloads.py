"""
Imports manually downloaded ElevenLabs/Poe clips and builds the training manifest.

Two modes:

1. FUZZY mode (default): matches filenames to training texts by text similarity.
   ElevenLabs names files after first ~50 chars; Poe uses full text stripped of punctuation.

2. INDEXED mode (--indexed): files named voice_01.mp3, voice_02.mp3, ...
   Maps directly to TEXTS[0], TEXTS[1], ... by number (1-based).
   Use this for ElevenLabs Projects exports or any manually numbered downloads.

Usage:
    # Fuzzy match a folder of text-named files
    python import_elevenlabs_downloads.py --folder "C:/Users/Owen/Downloads" --voice rachel

    # Indexed mode: arabella_01.mp3 -> TEXTS[0], arabella_02.mp3 -> TEXTS[1], ...
    python import_elevenlabs_downloads.py --folder "C:/Users/Owen/Downloads" --voice arabella --indexed

    # Preview without writing anything
    python import_elevenlabs_downloads.py --folder "..." --voice arabella --indexed --dry-run

    # Auto-scan data/paralinguistic/downloads/ subfolders (fuzzy)
    python import_elevenlabs_downloads.py
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

def normalize(s: str) -> str:
    """Normalize string the way Poe generates filenames:
    - Remove [brackets] but keep tag words
    - Remove all punctuation
    - Collapse whitespace, lowercase
    Works for both filenames and corpus texts.
    """
    s = Path(s).stem if '.' in s[-5:] else s   # strip extension if present
    s = re.sub(r'\[([^\]]+)\]', r'\1', s)       # [tag] -> tag
    s = re.sub(r"[^\w\s]", ' ', s)              # strip punctuation
    s = re.sub(r'[_\-]+', ' ', s)               # _ - -> space
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def best_match(filename: str, texts: list[str], threshold: float) -> tuple[int, float] | None:
    """Return (index, score) of the best-matching text, or None if below threshold.
    Handles both ElevenLabs (truncated text) and Poe (full text, no punctuation) filenames.
    """
    norm_fn = normalize(filename)
    best_i, best_score = -1, 0.0
    for i, text in enumerate(texts):
        norm_text = normalize(text)
        # Full comparison
        score = similarity(norm_fn, norm_text)
        # Also try matching filename against truncated text (ElevenLabs truncates at ~80 chars)
        score2 = similarity(norm_fn, norm_text[:len(norm_fn) + 15])
        score = max(score, score2)
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

def import_folder_indexed(folder: Path, voice_name: str, texts: list[str],
                          dry_run: bool) -> list[dict]:
    """Indexed mode: files named voicename_01.mp3, voicename_02.mp3, ...
    Maps file N directly to TEXTS[N-1]. Extra files beyond len(texts) are skipped.
    Missing files (gaps in numbering) are reported.
    """
    audio_exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
    # Only grab files that start with voice_name_ (case-insensitive)
    prefix = voice_name.lower() + "_"
    files = sorted([f for f in folder.iterdir()
                    if f.suffix.lower() in audio_exts
                    and f.stem.lower().startswith(prefix)])

    if not files:
        print(f"  [!] No audio files found in {folder} matching prefix '{prefix}'")
        return []

    # Extract numeric index from filename: voice_01.mp3 -> 1
    import re as _re
    def get_index(f: Path) -> int | None:
        m = _re.search(r'_(\d+)$', f.stem)
        return int(m.group(1)) if m else None

    indexed = {}
    for f in files:
        idx = get_index(f)
        if idx is not None and 1 <= idx <= len(texts):
            indexed[idx] = f
        elif idx is not None and idx > len(texts):
            print(f"  [skip] {f.name} (index {idx} > {len(texts)} texts)")

    print(f"\n  {len(files)} files found, {len(indexed)} within text range ({len(texts)} texts)")

    out_voice = OUT_DIR / voice_name.lower()
    if not dry_run:
        out_voice.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for ti in range(len(texts)):
        idx = ti + 1  # 1-based
        if idx not in indexed:
            missing.append(ti)
            continue

        f     = indexed[idx]
        text  = texts[ti]
        tags  = extract_tags(text)
        plain = strip_tags(text)

        dst_wav = out_voice / f"{ti:04d}_{voice_name.lower()}.wav"
        dst_txt = out_voice / f"{ti:04d}_{voice_name.lower()}.txt"

        status = "DRY RUN" if dry_run else "OK"
        print(f"  [{status}] {f.name:<40} -> {ti:04d}  tags={tags}")

        if not dry_run:
            ok = convert_to_wav(f, dst_wav)
            if ok:
                dst_txt.write_text(text, encoding="utf-8")
                rows.append({
                    "audio_path":  str(dst_wav),
                    "text":        text,
                    "plain_text":  plain,
                    "voice":       voice_name.lower(),
                    "tags":        "|".join(tags),
                    "match_score": "1.000",
                })
        else:
            rows.append({
                "audio_path":  str(dst_wav),
                "text":        text,
                "plain_text":  plain,
                "voice":       voice_name.lower(),
                "tags":        "|".join(tags),
                "match_score": "1.000",
            })

    if missing:
        print(f"\n  Missing indices ({len(missing)}): {missing[:20]}{'...' if len(missing)>20 else ''}")
        print(f"  These clips have no file — they will be skipped from the manifest.")

    print(f"\n  Imported: {len(rows)}/{len(texts)} for voice '{voice_name}'")
    return rows


def import_folder(folder: Path, voice_name: str, texts: list[str],
                  threshold: float, dry_run: bool, poe_suffix: int | None = None) -> list[dict]:
    """Match all audio files in folder to texts, convert + rename, return manifest rows.

    Uses optimal conflict resolution: scores ALL files against ALL texts first,
    then assigns highest-score pairs greedily so music files never steal slots
    from real training clips.

    poe_suffix: if set, only include files ending with ' (N)' in their stem (Poe duplicate suffix).
                Use 0 for base files (no suffix), 1 for '(1)', etc.
    """
    import re as _re
    audio_exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
    all_files = [f for f in folder.iterdir() if f.suffix.lower() in audio_exts]

    if poe_suffix is not None:
        def has_suffix(f: Path, n: int) -> bool:
            if n == 0:
                return not bool(_re.search(r'\s*\(\d+\)$', f.stem))
            return bool(_re.search(rf'\s*\({n}\)$', f.stem))
        files = [f for f in all_files if has_suffix(f, poe_suffix)]
        print(f"  Filtering to Poe suffix {'(base)' if poe_suffix==0 else f'({poe_suffix})'}: {len(files)}/{len(all_files)} files")
    else:
        files = all_files

    if not files:
        print(f"  [!] No audio files found in {folder}")
        return []

    print(f"\n  {len(files)} files found in {folder.name}/")

    # Score every file against every text
    scores = []  # (score, file_idx, text_idx)
    for fi, f in enumerate(files):
        for ti, text in enumerate(texts):
            s = similarity(normalize(f.stem), normalize(text))
            if s >= threshold:
                scores.append((s, fi, ti))

    # Sort by score descending, greedily assign best matches
    scores.sort(reverse=True)
    assigned_files = {}  # file_idx -> (text_idx, score)
    assigned_texts = {}  # text_idx -> (file_idx, score)
    for s, fi, ti in scores:
        if fi not in assigned_files and ti not in assigned_texts:
            assigned_files[fi] = (ti, s)
            assigned_texts[ti] = (fi, s)

    out_voice = OUT_DIR / voice_name.lower()
    if not dry_run:
        out_voice.mkdir(parents=True, exist_ok=True)

    rows = []
    for fi, f in enumerate(files):
        name_safe = f.name.encode('ascii', errors='replace').decode('ascii')
        if fi not in assigned_files:
            print(f"  [skip]    {name_safe[:65]}")
            continue

        ti, score = assigned_files[fi]
        text  = texts[ti]
        tags  = extract_tags(text)
        plain = strip_tags(text)

        dst_wav = out_voice / f"{ti:04d}_{voice_name.lower()}.wav"
        dst_txt = out_voice / f"{ti:04d}_{voice_name.lower()}.txt"

        status = "DRY RUN" if dry_run else "OK"
        print(f"  [{status}] {name_safe[:55]}")
        print(f"         -> {ti:04d}  score={score:.2f}  tags={tags}")

        if not dry_run:
            ok = convert_to_wav(f, dst_wav)
            if ok:
                dst_txt.write_text(text, encoding="utf-8")
                rows.append({
                    "audio_path":  str(dst_wav),
                    "text":        text,
                    "plain_text":  plain,
                    "voice":       voice_name.lower(),
                    "tags":        "|".join(tags),
                    "match_score": f"{score:.3f}",
                })
        else:
            rows.append({
                "audio_path":  str(dst_wav),
                "text":        text,
                "plain_text":  plain,
                "voice":       voice_name.lower(),
                "tags":        "|".join(tags),
                "match_score": f"{score:.3f}",
            })

    matched_idxs = {int(r["audio_path"].replace("\\","/").split("/")[-1][:4]) for r in rows}
    unmatched    = [i for i in range(len(texts)) if i not in matched_idxs]
    if unmatched:
        print(f"\n  Missing texts ({len(unmatched)}): indices {unmatched}")
        print(f"  These clips weren't found — check Downloads for those files.")

    print(f"\n  Matched: {len(rows)}/{len(texts)} texts for voice '{voice_name}'")
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
    parser.add_argument("--indexed",    action="store_true",
                        help="Use indexed mode: voice_01.mp3 -> TEXTS[0], voice_02.mp3 -> TEXTS[1], ...")
    parser.add_argument("--poe-suffix", type=int, default=None,
                        help="Only import Poe files with this Windows duplicate suffix (0=base, 1=(1), 2=(2), ...)")
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
        if args.indexed:
            rows = import_folder_indexed(folder, voice_name, TEXTS, args.dry_run)
        else:
            rows = import_folder(folder, voice_name, TEXTS, args.threshold, args.dry_run, args.poe_suffix)
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
