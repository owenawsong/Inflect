#!/usr/bin/env python3
r"""
Generate a VoxCPM2 cloning dataset locally from a prebuilt generation plan.

This is the path meant for rental GPUs. It does not use the public HF Space.

Single-GPU example:
  .\.venv-voxcpm\Scripts\python.exe scripts\generate_voxcpm_dataset_local.py ^
    --plan-file outputs\corpora\voxcpm_generation_plan_v2_60k.csv ^
    --version 20260412_voxcpm_v2_local ^
    --device-id 0

Multi-GPU sharding example (run one process per GPU):
  shard 0: --device-id 0 --shard-id 0 --num-shards 4
  shard 1: --device-id 1 --shard-id 1 --num-shards 4
  shard 2: --device-id 2 --shard-id 2 --num-shards 4
  shard 3: --device-id 3 --shard-id 3 --num-shards 4
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT_ROOT / "outputs" / "corpora" / "voxcpm_generation_plan_v2_60k.csv"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "voxcpm_dataset"
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


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_completed_ids(version_dir: Path) -> set[str]:
    done: set[str] = set()
    for csv_path in sorted(version_dir.glob("metadata*.csv")):
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    plan_id = (row.get("plan_id") or "").strip()
                    if plan_id:
                        done.add(plan_id)
        except FileNotFoundError:
            continue
    return done


def ensure_manifest(path: Path) -> None:
    if path.exists():
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_COLS)


def append_row(path: Path, row: dict[str, str]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        writer.writerow(row)


def write_readme(version_dir: Path, plan_file: Path, total_rows: int, shard_mode: bool) -> None:
    readme = version_dir / "README.md"
    card = f"""---
license: other
task_categories:
  - text-to-speech
language:
  - en
pretty_name: Inflect VoxCPM2 Local Synthetic Dataset
---

# Inflect VoxCPM2 Local Synthetic Dataset

- Source model: `openbmb/VoxCPM2`
- Plan file: `{plan_file.name}`
- Planned clips: `{total_rows}`
- Mode: ultimate cloning only
- Sharded metadata: `{shard_mode}`

This dataset was generated locally from a prebuilt text-to-voice plan.

Important licensing note:

- this release contains generated outputs only
- prompt-reference audio is not included
- the dataset uses `license: other` because the dataset should not inherit the repo code license automatically
"""
    readme.write_text(card, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-file", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--version", type=str, required=True)
    ap.add_argument("--model", default="openbmb/VoxCPM2")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--inference-timesteps", type=int, default=10)
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--denoise-reference", action="store_true")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--max-runtime-seconds", type=int, default=None)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    if args.num_shards < 1 or args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit("Invalid shard configuration.")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    from voxcpm import VoxCPM

    version_dir = args.out / args.version
    audio_dir = version_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_name = "metadata.csv" if args.num_shards == 1 else f"metadata.shard{args.shard_id:02d}.csv"
    manifest_path = version_dir / manifest_name
    ensure_manifest(manifest_path)

    plan_rows = load_rows(args.plan_file)
    write_readme(version_dir, args.plan_file, len(plan_rows), shard_mode=args.num_shards > 1)
    completed = load_completed_ids(version_dir)

    shard_rows = [row for idx, row in enumerate(plan_rows) if idx % args.num_shards == args.shard_id]
    pending_rows = [row for row in shard_rows if row["plan_id"] not in completed]

    print(f"Device: cuda:{args.device_id}")
    print(f"Version: {args.version}")
    print(f"Plan rows in shard: {len(shard_rows)}")
    print(f"Pending rows: {len(pending_rows)}")
    print(f"Manifest: {manifest_path}")

    model = VoxCPM.from_pretrained(
        args.model,
        optimize=args.optimize,
        load_denoiser=args.denoise_reference,
    )
    sample_rate = model.tts_model.sample_rate

    start = time.time()
    done = 0
    stopped_by_time_limit = False

    for idx, row in enumerate(pending_rows, start=1):
        if args.max_runtime_seconds is not None and done > 0:
            elapsed = time.time() - start
            if elapsed >= args.max_runtime_seconds:
                stopped_by_time_limit = True
                break
        wav = model.generate(
            text=row["text"],
            reference_wav_path=row["reference_wav"],
            prompt_wav_path=row["reference_wav"],
            prompt_text=row["prompt_text"] or None,
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
            normalize=args.normalize,
            denoise=args.denoise_reference,
        )
        out_name = f"{row['voice_id']}_ultimate_{row['plan_id']}.wav"
        out_path = audio_dir / out_name
        sf.write(str(out_path), wav, sample_rate)
        append_row(
            manifest_path,
            {
                "plan_id": row["plan_id"],
                "file_name": f"audio/{out_name}",
                "text": row["text"],
                "category": row["category"],
                "voice_id": row["voice_id"],
                "reference_wav": row["reference_wav"],
                "prompt_text": row["prompt_text"],
                "cfg_value": str(args.cfg_value),
                "inference_timesteps": str(args.inference_timesteps),
                "duration_s": f"{len(wav) / sample_rate:.3f}",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        done += 1
        if done % 25 == 0:
            elapsed = time.time() - start
            rate = done / elapsed * 3600 if elapsed > 0 else 0.0
            print(f"[{done}/{len(pending_rows)}] {rate:.0f} clips/hr")
        if args.max_clips and done >= args.max_clips:
            break
        if args.max_runtime_seconds is not None:
            elapsed = time.time() - start
            if elapsed >= args.max_runtime_seconds:
                stopped_by_time_limit = True
                break

    elapsed = time.time() - start
    rate = done / elapsed * 3600 if elapsed > 0 else 0.0
    if stopped_by_time_limit:
        print(f"Reached max runtime of {args.max_runtime_seconds}s. Exiting cleanly.")
    print(f"Done. {done} new clips in {elapsed/3600:.2f}h ({rate:.0f} clips/hr)")
    print(f"Output: {version_dir}")


if __name__ == "__main__":
    main()
