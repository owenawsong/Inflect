#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Upload a VoxCPM dataset folder in AudioFolder format to the Hugging Face Dataset Hub."
    )
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory containing metadata.csv, README.md, and audio/.",
    )
    ap.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repo id, e.g. username/voxcpm2-synthetic-v1",
    )
    ap.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it does not already exist.",
    )
    ap.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN env vars.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print what would be uploaded without performing network actions.",
    )
    return ap.parse_args()


def require_path(path: Path, kind: str) -> None:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Missing required directory: {path}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    metadata = dataset_dir / "metadata.csv"
    readme = dataset_dir / "README.md"
    audio_dir = dataset_dir / "audio"

    require_path(metadata, "file")
    require_path(readme, "file")
    require_path(audio_dir, "dir")

    audio_count = sum(1 for _ in audio_dir.rglob("*.wav"))
    print(f"Dataset dir: {dataset_dir}")
    print(f"Repo id:     {args.repo_id}")
    print(f"Audio files: {audio_count}")
    print(f"Private:     {args.private}")

    if args.dry_run:
        print("Dry run complete. Inputs look valid.")
        return

    if not token:
        raise RuntimeError(
            "No Hugging Face token provided. Set HF_TOKEN or pass --token."
        )

    api = HfApi(token=token)
    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(dataset_dir),
        path_in_repo=".",
        ignore_patterns=[
            "**/__pycache__/**",
            "*.tmp",
        ],
    )

    print(f"Upload complete: {repo_url}")


if __name__ == "__main__":
    main()
