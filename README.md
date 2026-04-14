# Inflect

Inflect is a local research workspace for building a stronger English zero-shot TTS stack around ZipVoice.

The repository is not a polished pip package yet. It is the actual working project used for:

- VoxCPM2 synthetic dataset generation
- ZipVoice and Lux-inspired runtime benchmarking
- ZipVoice teacher fine-tuning
- Inflect-specific enhancement and paralinguistic experiments

## Current Status

- Main backbone direction: `ZipVoice`
- Current runtime default: `inflect_base + lux solver`
- Current synthetic foundation dataset: VoxCPM2 English corpus
- Current fine-tune target: ZipVoice teacher adapted on VoxCPM data
- Current phase: foundation fine-tuning and dataset scaling

This repo already contains real operational scripts, but parts of the project are still research-grade rather than release-grade.

## Current Runtime Stack

The currently chosen inference stack is:

- `zipvoice_distill`
- `4` inference steps
- guidance scale `3.0`
- direct prompt transcripts
- Lux/LinaCodec-inspired `48k` decoder path
- Lux solver
- `t_shift = 0.9`
- target RMS `0.01`
- no Lux `speed * 1.3` hack
- no default prompt cap

That is the working Inflect base today. Real LuxTTS is kept as a benchmark and idea source, not the training base.

## What Is In This Repo

### `scripts/`

Operational entrypoints for:

- VoxCPM HF Space generation
- VoxCPM local rental-GPU generation
- text-corpus construction
- generation-plan construction
- ZipVoice/Lux/Inflect comparison benches
- teacher fine-tuning prep and launch
- dataset upload and release prep

See [scripts/README.md](C:/users/owen/Inflect-New/scripts/README.md).

### `inflect/`

Inflect-native model code and experiments:

- `enhancer/`
- `para_module/`
- `data/`

See [inflect/README.md](C:/users/owen/Inflect-New/inflect/README.md).

### `voice-encoder/`

Separate voice-conditioning and paralinguistic research area with its own training and preprocessing scripts.

See [voice-encoder/README.md](C:/users/owen/Inflect-New/voice-encoder/README.md).

### `CLAUDE_READ/`

Human-readable project notes, roadmap, and current execution state.

Start with:

- [claude.md](C:/users/owen/Inflect-New/CLAUDE_READ/claude.md)
- [plan.md](C:/users/owen/Inflect-New/CLAUDE_READ/plan.md)
- [todo.md](C:/users/owen/Inflect-New/CLAUDE_READ/todo.md)

### Local external dependencies

These are used locally but are intentionally not committed as part of the main GitHub repo:

- `ZipVoice-official/`
- `third_party/`
- `reference_voices/`
- `outputs/`

The repo stores the local ZipVoice fixes as a patch file:

- [zipvoice-local-fixes.patch](C:/users/owen/Inflect-New/patches/zipvoice-local-fixes.patch)

## Main Workflows

### 1. Build VoxCPM2 synthetic data through the HF Space

```powershell
cd C:\Users\Owen\Inflect-New
.\.venv-voxcpm\Scripts\python.exe scripts\generate_voxcpm_dataset.py --workers 2 --version 20260411_large_text_v1 --corpus-file outputs\corpora\voxcpm_texts_20000_v2.csv --max-clips 60000
```

### 2. Build the same dataset locally on a rental GPU

```powershell
cd C:\Users\Owen\Inflect-New
.\.venv-voxcpm\Scripts\python.exe scripts\generate_voxcpm_dataset_local.py --plan-file outputs\corpora\voxcpm_generation_plan_v2_60k.csv --version 20260412_voxcpm_v2_local --device-id 0 --optimize --normalize
```

### 3. Run ZipVoice / Inflect comparisons

```powershell
cd C:\Users\Owen\Inflect-New
.\.venv-voxcpm\Scripts\python.exe scripts\run_inflect_base_bench.py
```

### 4. Launch teacher fine-tuning

```powershell
cd C:\Users\Owen\Inflect-New
.\.venv-voxcpm\Scripts\python.exe scripts\run_inflect_teacher_finetune.py --preset demo --work-dir outputs\inflect_finetune\teacher_foundation_demo
```

## Dataset Status

The current active VoxCPM dataset run is:

- `outputs/voxcpm_dataset/20260411_large_text_v1`

Current generated size at the time of this documentation update:

- `21,543` clips
- `56` voices
- mixed old and `v2` text pools

Planned clean `v2` target:

- `20,000` English texts
- `60,000` clips
- `56` voices
- about `103` hours total

Important files:

- [voxcpm_texts_20000_v2.csv](C:/users/owen/Inflect-New/outputs/corpora/voxcpm_texts_20000_v2.csv)
- [voxcpm_generation_plan_v2_60k.csv](C:/users/owen/Inflect-New/outputs/corpora/voxcpm_generation_plan_v2_60k.csv)

## Fine-Tuning Status

The teacher fine-tuning path is now working locally again.

Recent fix:

- the non-finite gradient crash was caused by the non-`k2` fallback Swoosh activation in ZipVoice
- the local fix is preserved in [zipvoice-local-fixes.patch](C:/users/owen/Inflect-New/patches/zipvoice-local-fixes.patch)

Current training intent:

- foundation fine-tune the ZipVoice teacher on VoxCPM synthetic data
- validate cloning, consistency, and quality improvements
- later add expressive data once the foundation pass is stable

## GitHub vs Hugging Face

This project is intentionally split:

- GitHub:
  - source code
  - docs
  - scripts
  - patch files
- Hugging Face Datasets:
  - generated VoxCPM synthetic datasets
- Local only:
  - checkpoints
  - outputs
  - reference voice audio
  - vendored external repos

See [PUBLISHING.md](C:/users/owen/Inflect-New/PUBLISHING.md).

## Licensing

Repository code and docs are licensed under Apache 2.0 unless otherwise noted.

Important exceptions:

- external code in local-only repos such as `ZipVoice-official/` and `third_party/` follows its own upstream licenses
- generated datasets are documented separately and should not inherit the repo code license automatically
- reference voice audio is not part of the public GitHub repo

See [LICENSE](C:/users/owen/Inflect-New/LICENSE) and [PUBLISHING.md](C:/users/owen/Inflect-New/PUBLISHING.md).
