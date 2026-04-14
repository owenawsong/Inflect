# Project Structure

## Core code

- `inflect/`
  - Inflect project code
- `scripts/`
  - operational scripts for generation, benchmarking, fine-tuning, and serving compare pages
- `ZipVoice-official/`
  - upstream ZipVoice training/inference code used for baseline and fine-tuning
- `third_party/`
  - vendored external code such as LuxTTS / LinaCodec integrations

## Data

- `reference_voices/`
  - active reference voice set for cloning
- `reference_voices_backup/`
  - backup copy of reference voices
- `outputs/`
  - generated artifacts, datasets, benches, debug audio, and training runs

## Important current files

- `outputs/corpora/voxcpm_texts_20000_v2.csv`
  - current `v2` text pool
- `outputs/corpora/voxcpm_generation_plan_v2_60k.csv`
  - current `60k` generation plan
- `scripts/generate_voxcpm_dataset.py`
  - HF Space generator
- `scripts/generate_voxcpm_dataset_local.py`
  - local rental-GPU generator
- `scripts/run_inflect_teacher_finetune.py`
  - fine-tuning launcher

## Practical rule

If a folder name is ambiguous, prefer checking:

- `outputs/README.md`
- `outputs/voxcpm_dataset/INDEX.md`
- `outputs/inflect_finetune/README.md`

before opening random metadata files.
