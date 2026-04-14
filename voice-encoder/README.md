# Voice Encoder Area

This folder is a separate research track for cloning support, reference conditioning, and paralinguistic experiments.

It is not the current primary backbone path, but it still contains useful source code.

## What lives here

- voice encoder training scripts
- adapter experiments
- paralinguistic data generation helpers
- manual/import utilities for external voice data

## Important scripts

- `scripts/train_ve.py`
- `scripts/train_1d_reference_conditioning.py`
- `scripts/train_phase3_paralinguistic.py`
- `scripts/import_elevenlabs_downloads.py`
- `scripts/generate_paralinguistic_data.py`

## Current status

- useful as a side research area
- not the current blocker
- should not distract from the ZipVoice teacher foundation work

## Data policy

Large generated artifacts and raw/mel datasets remain local only.

This repo should publish:

- source code
- small docs
- reproducible scripts

It should not publish:

- giant local training artifacts
- private voice data
- generated checkpoint dumps
