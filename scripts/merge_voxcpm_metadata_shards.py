#!/usr/bin/env python3
"""
Merge sharded local VoxCPM metadata files into metadata.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MANIFEST_COLS = [
    "plan_id",
    "file_name",
    "text",
    "category",
    "voice_id",
    "reference_wav",
    "prompt_text",
    "cfg_value",
    "inference_timesteps",
    "duration_s",
    "generated_at",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version-dir", type=Path, required=True)
    args = ap.parse_args()

    shard_files = sorted(args.version_dir.glob("metadata.shard*.csv"))
    if not shard_files:
        raise SystemExit("No metadata.shard*.csv files found.")

    rows = []
    seen = set()
    for shard in shard_files:
        with open(shard, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                plan_id = row.get("plan_id", "")
                if plan_id and plan_id not in seen:
                    seen.add(plan_id)
                    rows.append(row)

    out_path = args.version_dir / "metadata.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
