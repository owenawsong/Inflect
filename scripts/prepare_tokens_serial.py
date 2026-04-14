#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lhotse import load_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
ZIPVOICE_DIR = REPO_ROOT / "ZipVoice-official"

# Avoid importing this repo's local `inflect/` package in place of the
# PyPI `inflect` dependency required by ZipVoice's text normalizer.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]
sys.path.insert(0, str(ZIPVOICE_DIR))

from zipvoice.tokenizer.tokenizer import add_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows-safe serial token preparation for ZipVoice manifests."
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        choices=["emilia", "espeak", "dialog", "libritts", "simple"],
        default="emilia",
    )
    parser.add_argument("--lang", type=str, default="en-us")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Loading manifest from %s", args.input_file)
    cut_set = load_manifest(args.input_file)
    logging.info("Adding tokens serially")
    cut_set = add_tokens(cut_set, tokenizer=args.tokenizer, lang=args.lang)
    logging.info("Saving manifest to %s", args.output_file)
    cut_set.to_file(args.output_file)
    logging.info("Done")


if __name__ == "__main__":
    main()
