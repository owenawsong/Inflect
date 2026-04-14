#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Freeze a VoxCPM dataset snapshot and rewrite its public-facing Hugging Face dataset card."
    )
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "voxcpm_dataset" / "20260411_large_text_v1",
        help="Source dataset directory containing metadata.csv and audio/.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "publish" / "voxcpm2_synthetic_en_v1_public",
        help="Destination snapshot directory.",
    )
    ap.add_argument(
        "--pretty-name",
        type=str,
        default="Inflect VoxCPM2 Synthetic English v1",
    )
    ap.add_argument(
        "--license-tag",
        type=str,
        default="other",
        help="Dataset card license tag. Defaults to 'other' because voice/source provenance is mixed.",
    )
    return ap.parse_args()


def require(path: Path, kind: str) -> None:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Missing required directory: {path}")


def load_rows(metadata_path: Path) -> list[dict[str, str]]:
    with open(metadata_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_readme(pretty_name: str, license_tag: str, rows: list[dict[str, str]]) -> str:
    voice_ids = sorted({row["voice_id"] for row in rows if row.get("voice_id")})
    categories = Counter(row["category"] for row in rows if row.get("category"))
    modes = Counter(row["mode"] for row in rows if row.get("mode"))
    top_categories = "\n".join(f"- `{name}`: {count}" for name, count in categories.most_common())
    top_modes = "\n".join(f"- `{name}`: {count}" for name, count in modes.most_common())
    voice_preview = ", ".join(voice_ids[:20]) + (", ..." if len(voice_ids) > 20 else "")

    return f"""---
license: {license_tag}
pretty_name: {pretty_name}
task_categories:
  - text-to-speech
language:
  - en
tags:
  - voxcpm2
  - tts
  - voice-cloning
  - synthetic
  - english
---

# {pretty_name}

Synthetic English speech dataset generated with [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).

## What this dataset is

- generated speech audio, not human-recorded source speech
- built for English TTS fine-tuning, cloning research, and benchmarking
- generated from a multi-voice prompt pool using VoxCPM2 ultimate-clone style prompting

## Current snapshot stats

| Field | Value |
|---|---|
| Clips | {len(rows)} |
| Distinct voices | {len(voice_ids)} |
| Languages | English only |
| Generation model | VoxCPM2 |
| Source layout | Hugging Face `AudioFolder` |

## Category distribution

{top_categories}

## Mode distribution

{top_modes}

## Voice pool

Distinct prompt-voice IDs in this snapshot: `{len(voice_ids)}`

Examples:

{voice_preview}

## Columns

| Column | Description |
|---|---|
| `file_name` | Relative audio path |
| `text` | Intended spoken text |
| `category` | Text bucket |
| `voice_id` | Prompt voice identifier |
| `mode` | Generation mode used for that sample |
| `speaker_gender` | Optional prompt metadata |
| `speaker_accent` | Optional prompt metadata |
| `speaker_style` | Optional prompt metadata |
| `control_instruction` | Voice-design instruction if used |
| `prompt_text` | Prompt transcript when available |
| `cfg_value` | CFG value used during generation |
| `duration_s` | Clip duration in seconds |
| `generated_at` | Local generation timestamp |

## Important limitations

- This dataset is synthetic.
- The reference prompt audio used during generation is **not** included in this release.
- Voice/source provenance is mixed, so this release is labeled with a cautious `license: other` tag rather than a broad open-content claim.
- Downstream users are responsible for checking whether their intended use complies with upstream model terms, prompt-source rights, and local law.

## Recommended use

- TTS fine-tuning
- cloning and similarity evaluation
- inference benchmarking
- synthetic-data ablation work

## Not included

- reference voice WAV files
- local generation scripts' private caches
- training checkpoints
"""


def build_license_notice() -> str:
    return """Inflect VoxCPM2 public dataset notice

This dataset contains synthetic speech outputs generated from a private/local prompt-voice pool using VoxCPM2.

Important points:

1. The repository code license does not automatically apply to this dataset.
2. The bundled audio files are generated outputs only; prompt-reference audio is not included.
3. Prompt-source rights, upstream model terms, and jurisdiction-specific rules may still matter for downstream use.
4. This release is intended for research, benchmarking, and model-development use where the user accepts responsibility for compliant use.

No additional rights to any omitted prompt-source audio, names, or third-party trademarks are granted by this notice.
"""


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()

    metadata = dataset_dir / "metadata.csv"
    audio_dir = dataset_dir / "audio"

    require(metadata, "file")
    require(audio_dir, "dir")

    rows = load_rows(metadata)
    if out_dir.exists():
        raise FileExistsError(f"Destination already exists: {out_dir}")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dataset_dir, out_dir)

    (out_dir / "README.md").write_text(
        build_readme(args.pretty_name, args.license_tag, rows),
        encoding="utf-8",
    )
    (out_dir / "LICENSE_NOTICE.txt").write_text(build_license_notice(), encoding="utf-8")

    print(f"Prepared public dataset snapshot: {out_dir}")
    print(f"Rows: {len(rows)}")
    print(f"Audio files: {sum(1 for _ in (out_dir / 'audio').rglob('*.wav'))}")


if __name__ == "__main__":
    main()
