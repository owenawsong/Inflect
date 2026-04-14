# Claude's Head — Inflect Project Notes

Last updated: 2026-04-13

## Current Truth

Inflect is currently a ZipVoice-centered project.

That means:

- ZipVoice is the backbone
- LuxTTS is a reference and benchmark source
- VoxCPM2 is the main synthetic foundation dataset
- Inflect-Enhance is downstream of the backbone, not the other way around

The working runtime default is:

- `zipvoice_distill`
- `4` steps
- guidance `3.0`
- direct transcripts
- Lux/LinaCodec-inspired `48k` decoder path
- Lux solver
- `t_shift = 0.9`
- target RMS `0.01`
- no copied Lux speed multiplier
- no default prompt cap

## What Changed Recently

### 1. Lux investigation was narrowed correctly

Lux is useful as a source of concrete ideas, mainly:

- solver update
- 48k decoder path
- wrapper/inference simplifications

But it is not the project base.

### 2. VoxCPM dataset work matured

Current planning assets:

- `outputs/corpora/voxcpm_texts_20000_v2.csv`
- `outputs/corpora/voxcpm_generation_plan_v2_60k.csv`

Current active HF Space dataset:

- `outputs/voxcpm_dataset/20260411_large_text_v1`

Current active size when this note was updated:

- `21,543` clips
- `56` voices

### 3. Teacher fine-tuning was unblocked

The crash was not caused by a bad early dataset batch.

The real cause was:

- non-`k2` fallback Swoosh activation in ZipVoice
- raw exponential form overflowed
- gradients became non-finite on the first training step

The local fix now lives in:

- `patches/zipvoice-local-fixes.patch`

And the current training launcher is:

- `scripts/run_inflect_teacher_finetune.py`

## Project Priorities

Priority order right now:

1. cloning accuracy
2. consistency
3. overall quality
4. long-form stability
5. emotional richness

Emotional richness matters, but it does not outrank cloning accuracy or stability yet.

## Dataset Strategy

### Foundation dataset

Use VoxCPM synthetic data first.

Why:

- strong voice similarity behavior
- strong stability
- good clean synthetic supervision
- useful for zero-shot cloning behavior

### Expressive dataset later

Add expressive data after the foundation pass works.

Current best candidate:

- Expresso

Reason:

- better style/prosody supervision than small acted-emotion corpora
- more relevant to expressive TTS than CREMA-D as a first expressive add-on

## Hard Rules

- do not make LuxTTS the base by convenience
- do not train Inflect-Enhance before the backbone is settled
- do not claim the repo is release-ready when it is still a research workspace
- do not call the dataset openly licensed unless its provenance supports that claim
- do not overfit the roadmap to one impressive sample

## Current Publish Direction

GitHub should become:

- source
- scripts
- docs
- patch files

Hugging Face should hold:

- the VoxCPM synthetic dataset snapshots

The public dataset should use a cautious release posture:

- clear card
- clear provenance note
- no fake `apache-2.0` dataset claim
