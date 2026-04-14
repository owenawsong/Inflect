#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare VoxCPM metadata into ZipVoice custom train/dev TSV files."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the dataset version directory containing metadata.csv and audio/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for custom_train.tsv, custom_dev.tsv, and stats files.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.02,
        help="Approximate validation split ratio within each voice bucket.",
    )
    parser.add_argument(
        "--min-dev-per-voice",
        type=int,
        default=4,
        help="Minimum number of validation samples per voice when possible.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting.",
    )
    return parser.parse_args()


def read_rows(dataset_dir: Path) -> list[dict]:
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata.csv: {metadata_path}")

    rows: list[dict] = []
    seen_files: set[str] = set()
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            text = (row.get("text") or "").strip()
            voice_id = (row.get("voice_id") or "unknown").strip() or "unknown"
            if not file_name or not text:
                continue
            if file_name in seen_files:
                continue
            wav_path = (dataset_dir / file_name).resolve()
            if not wav_path.is_file():
                continue
            seen_files.add(file_name)
            row["voice_id"] = voice_id
            row["text"] = text
            row["abs_wav_path"] = str(wav_path)
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No usable rows found in {metadata_path}")
    return rows


def choose_dev_rows(
    rows: list[dict], dev_ratio: float, min_dev_per_voice: int, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_voice: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_voice[row["voice_id"]].append(row)

    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    for voice_id in sorted(by_voice):
        items = by_voice[voice_id][:]
        rng.shuffle(items)

        if len(items) <= 1:
            train_rows.extend(items)
            continue

        target_dev = max(min_dev_per_voice, round(len(items) * dev_ratio))
        target_dev = min(target_dev, len(items) - 1)
        dev_rows.extend(items[:target_dev])
        train_rows.extend(items[target_dev:])

    return train_rows, dev_rows


def make_uniq_id(row: dict) -> str:
    return Path(row["file_name"]).stem


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for row in rows:
            writer.writerow([make_uniq_id(row), row["text"], row["abs_wav_path"]])


def write_stats(path: Path, rows: list[dict], train_rows: list[dict], dev_rows: list[dict]) -> None:
    voice_counts: dict[str, int] = defaultdict(int)
    category_counts: dict[str, int] = defaultdict(int)
    total_duration = 0.0

    for row in rows:
        voice_counts[row["voice_id"]] += 1
        category_counts[(row.get("category") or "unknown").strip() or "unknown"] += 1
        try:
            total_duration += float(row.get("duration_s") or 0.0)
        except ValueError:
            pass

    stats = {
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "total_duration_hours": round(total_duration / 3600.0, 3),
        "num_voices": len(voice_counts),
        "voices": dict(sorted(voice_counts.items())),
        "categories": dict(sorted(category_counts.items())),
    }
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    rows = read_rows(args.dataset_dir.resolve())
    train_rows, dev_rows = choose_dev_rows(
        rows=rows,
        dev_ratio=args.dev_ratio,
        min_dev_per_voice=args.min_dev_per_voice,
        seed=args.seed,
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    write_tsv(out_dir / "custom_train.tsv", train_rows)
    write_tsv(out_dir / "custom_dev.tsv", dev_rows)
    write_stats(out_dir / "split_stats.json", rows, train_rows, dev_rows)

    print(f"Prepared train/dev TSVs in: {out_dir}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Dev rows:   {len(dev_rows)}")
    print(f"Stats:      {out_dir / 'split_stats.json'}")


if __name__ == "__main__":
    main()
