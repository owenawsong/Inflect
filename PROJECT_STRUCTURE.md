# Project Structure

This file describes the publishable project layout. Local generated artifacts, checkpoints, and third-party checkouts are intentionally excluded from GitHub.

## Top level

- `README.md`
  - public-facing project overview
- `LICENSE`
  - Apache 2.0 for repo code/docs unless otherwise noted
- `PUBLISHING.md`
  - GitHub and Hugging Face release guidance
- `PROJECT_STRUCTURE.md`
  - this file

## Main source folders

### `scripts/`

Operational scripts for the active workflows:

- VoxCPM HF Space generation
- VoxCPM local generation on rental GPUs
- corpus and generation-plan building
- runtime benchmarks and comparison servers
- teacher fine-tune preparation and launch
- dataset upload and release prep

Use [scripts/README.md](scripts/README.md) as the real entrypoint.

### `inflect/`

Inflect-specific model and experiment code:

- `enhancer/`
  - audio enhancement pipeline
- `para_module/`
  - paralinguistic sound generation experiments
- `data/`
  - dataset utilities, extraction helpers, tag definitions

Use [inflect/README.md](inflect/README.md).

### `voice-encoder/`

Separate research area for:

- reference conditioning
- cloning support
- paralinguistic data prep
- adapter training experiments

Use [voice-encoder/README.md](voice-encoder/README.md).

### `CLAUDE_READ/`

Living project notes:

- `claude.md`
  - current truth and decisions
- `plan.md`
  - roadmap
- `todo.md`
  - execution checklist
- `lux_*`
  - Lux investigation notes
- `zipvoice_experiment_spec.md`
  - benchmark design

## Local-only folders

These are intentionally not part of the publishable GitHub payload.

### `outputs/`

Generated artifacts:

- datasets
- corpora
- benches
- fine-tune runs
- debug audio

### `reference_voices/`

Reference voice prompts used for cloning and VoxCPM generation.

### `ZipVoice-official/`

Local upstream checkout of official ZipVoice used for:

- baseline inference
- training recipes
- local teacher fine-tuning

### `third_party/`

Other local external repos and integrations such as LuxTTS/LinaCodec experiments.

## Important current files

- [run_inflect_teacher_finetune.py](scripts/run_inflect_teacher_finetune.py)
  - main launcher for ZipVoice teacher fine-tuning
- [generate_voxcpm_dataset.py](scripts/generate_voxcpm_dataset.py)
  - HF Space dataset generation
- [generate_voxcpm_dataset_local.py](scripts/generate_voxcpm_dataset_local.py)
  - local/rental-GPU dataset generation
- [build_voxcpm_text_corpus_v2.py](scripts/build_voxcpm_text_corpus_v2.py)
  - current higher-quality English corpus builder
- [build_voxcpm_generation_plan.py](scripts/build_voxcpm_generation_plan.py)
  - current balanced 60k plan builder
- [zipvoice-local-fixes.patch](patches/zipvoice-local-fixes.patch)
  - required local patch set for current teacher fine-tuning

## What Not To Assume

- This repo is not a single clean packaged library yet.
- Some subareas are active, some are parked.
- Runtime benchmarking is much more mature than the paralinguistic branch.
- The public GitHub repo should contain source and docs, not giant local artifacts.
