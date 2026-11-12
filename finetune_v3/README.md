# Inflect Mini v3 Adaptation

Prepare, audit, train, export, and run a separate native24 Mini adaptation using
your own licensed recordings and **externally aligned phone durations**. The
tooling is mechanically tested; useful new-language quality still requires a
representative pilot and fluent-speaker evaluation. No Polish-quality result
is claimed. This is not an automatic aligner or the v2 VITS trainer; Micro and
Atom are not supported.

Start with the [language-adaptation guide](../docs/v3/language-adaptation/index.md)
or go directly to [command recipes](../docs/v3/language-adaptation/commands.md).

## 0.1.2 Training Execution Profile

Version0.1.2 expresses the existing reflection values through explicit
slice/flip/concatenation in the training bridge and spectral-loss path. Public
inference classes, source weights and the serialized inference schema remain
unchanged. The source-import and exact mapped-phone protections remain in place.

New training contracts pin `execution_profile=native24-reflect-cat-v1`.
Resume requires that exact profile, the complete original contract and matching
checkpoint/controller sidecars. Missing, older or altered profiles reject before
model/checkpoint loading; there is no cross-version optimizer-resume waiver.
Do not edit old contracts to bypass this check. Valid legacy checkpoints may
still be exported under their original verified contracts, or exported weights
may initialize a new run; neither operation resumes the old optimizer.

See [version-specific qualification](QUALIFICATION.md) for actual tested groups,
environment and limits. Speaker-only CUDA evidence must not be generalized to
the default four-group configuration. The historical checks below keep their
original version/device scope.

## 0.1.1 Release Candidate

Historical version 0.1.1 adds source-only verified model imports without deleting existing
bytecode caches, and exact alignment binding for explicitly mapped phone
codepoints. See the [changelog](CHANGELOG.md) and
[security boundary](../docs/v3/language-adaptation/security-notes.md).
The full CPU suite passed **199 tests, 1 skipped, 23 warnings**, including
42 real-model cases. Version 0.1.1 also passed offline isolated-target import,
strict checkpoint/export and raw-text inference from an unrelated Temp working
directory: mono24 PCM16, 120 native frames, using existing qualified dependencies.
This is not a fresh complete-dependency installation, a new CUDA-training
qualification or a new-language quality result. Exact artifact pins and scope
are recorded in [tested coverage](../docs/v3/language-adaptation/implementation-status.md).

## Install and Obtain the Base

Use Python 3.12 and the appropriate PyTorch CPU/CUDA dependencies. From the
repository root:

```sh
python -m pip install './finetune_v3'
python -m finetune_v3 --help
```

The [tested dependency snapshot](requirements-tested.txt) records versions from
the existing verification environments, not a complete fresh-environment lock.
Install the appropriate CPU or CUDA PyTorch build separately for your machine.

The installed entry point is also `inflect-v3-adapt`. Historical version **0.1.0**, wheel
build **r004**, passed raw-text generation from an unrelated Temp working
directory using an isolated `--target` installation and an existing qualified
dependency environment: mono24, 193 native frames. This is not a freshly
downloaded complete-environment test.

For complete pinned acquisition and verification, follow the [full-base recipe](../docs/v3/language-adaptation/pinned-base.md). For consistent phone/vocabulary/migration files and the separate phoneme-input/raw-text workflows, see [coordinated vocabulary](../docs/v3/language-adaptation/coordinated-vocabulary.md) and [two input workflows](../docs/v3/language-adaptation/two-input-workflows.md).

Obtain the full official Mini package separately; the examples use
`./inflect-mini-v3`. The tested public model-core `weights.pt` SHA256 is:

```text
8460e7d54a2fbf7fb54b3c7be7569c493c852de9a2aa1a0fd9fd9a099de581ed
```

Retain the package's root license documents, core notices, and `core/LICENSES/`
texts: CLI export requires them, not only a bare model directory. The toolkit
includes its Apache `LICENSE` and setuptools-compatible license-file metadata.

## Adapt Your Data

1. Extract the base map with `vocab`, then use `coverage` to inspect your
   language/frontend output without needing audio or alignment labels.
2. Collect consenting-speaker recordings, verify transcripts, and obtain real
   external phone timing. Prepare grouped train/validation data and run `audit`.
3. Train a representative pilot, select a checkpoint on held-out evidence,
   then `export` and `say` from the separate adapted artifact.

Prepared audio is mono24 with exactly `600 * sum(phone_durations)` samples.
Positive integer counts include both boundary tokens. No guessed uniform labels
or base-model estimates for unrelated recordings are accepted. See the
[data contract](../docs/v3/language-adaptation/workflow.md#prepared-data-contract).

Training uses full unpadded batch-one utterances. Initial groups are
`text_encoder duration pitch_noise speaker`, with bridge/generator frozen;
`duration` includes shared prosody context. The objective combines supervised
`log1p` duration and multi-resolution waveform spectral loss; no F0 extraction
is required. New phones need an explicit reviewed unused-ID migration, not
reassignment of existing IDs. Pilot settings depend on your corpus, not the
tiny mechanical smoke below.

## Run the Mechanical Smoke

The fixture contains six self-generated English Cassidy examples and their
exact synthesis clocks. It tests plumbing, not alignment of human recordings
or foreign-language quality. The four-group CUDA commands below are illustrative,
not a qualified0.1.2 default-group recipe. Do not infer admission or successful
execution from these examples; check [version-specific scope](QUALIFICATION.md).
Use new output names outside the source package:

```sh
mkdir adaptation-work
python -B -m finetune_v3 smoke-dataset --model ./inflect-mini-v3 --output ./adaptation-work/fixture --acknowledge-mechanical-only
python -B -m finetune_v3 audit --dataset ./adaptation-work/fixture/prepared/dataset.json
python -B -m finetune_v3 train --model ./inflect-mini-v3 --dataset ./adaptation-work/fixture/prepared/dataset.json --output ./adaptation-work/run --voice cassidy --device cuda --groups text_encoder duration pitch_noise speaker --steps 2 --learning-rate 0.00001 --gradient-accumulation 1 --checkpoint-every 2 --validation-every 2 --max-validation-clips 1 --max-frames 120 --seed 2026 --threads 2
python -B -m finetune_v3 train --model ./inflect-mini-v3 --dataset ./adaptation-work/fixture/prepared/dataset.json --output ./adaptation-work/run --voice cassidy --device cuda --groups text_encoder duration pitch_noise speaker --steps 4 --learning-rate 0.00001 --gradient-accumulation 1 --checkpoint-every 2 --validation-every 2 --max-validation-clips 1 --max-frames 120 --seed 2026 --threads 2 --resume ./adaptation-work/run/checkpoints/step-0000002.pt
python -B -m finetune_v3 export --run ./adaptation-work/run --checkpoint ./adaptation-work/run/checkpoints/step-0000004.pt --output ./adaptation-work/export
python -B -m finetune_v3 say --model ./adaptation-work/export --phonemes "həlˈO." --output ./adaptation-work/hello.wav
```

CUDA must be available. For a separate CPU smoke use `--device cpu` consistently.
`--steps 4` is the final update target; resume/export need the matching controller
sidecar and run contract. The eSpeak variant uses the same fixed English prompts:

```sh
python -B -m finetune_v3 smoke-dataset --model ./inflect-mini-v3 --output ./adaptation-work/fixture-espeak --frontend espeak --acknowledge-mechanical-only
```

Train/export that fixture under separate output names to exercise raw-text
`say`. Historical 0.1.0 verification passed the eSpeak `(1, 52, 0)` CUDA
two-update/resume-four and continuous-four runs, with small nondeterminism,
not bit-exact equivalence. Historical 0.1.1 installed-package CPU export/raw-text
checks passed with the saved default Cassidy target and 120-frame mono24 output;
they do not constitute a new CUDA-training qualification. See
[tested scope and gates](../docs/v3/language-adaptation/implementation-status.md).

Historical 0.1.0 parent verification, including real-model integration and the notice
continuation regression, reported **180 passed, 1 skipped, 22 warnings in
104.14 seconds**. The separate lightweight run
reported 138 passed/1 skipped; v2 regression reported 39 passed. Source and
published weights were unchanged. These are software checks, not new-language
quality results. Installed-wheel evidence above applies to r004: 16 files,
including the Apache LICENSE, with no model or audio assets. Its artifact pins remain in the version-specific guide evidence.

## Export and Evaluate

Export validates checkpoint/controller/contract identity, records one
`target_voice`, and retains manifest-hashed Apache, StyleTTS2-MIT, and other
source notices without relicensing upstream material. `say` defaults to the
target voice and rejects unqualified slots. Prephonemized exports require
`--phonemes`; raw-text eSpeak exports require the saved engine version and
language resources, with no English fallback.

The artifact is an adapted core plus frontend metadata/notices, not the original
events, enhancer, or tuner package. Before sharing it, evaluate held-out raw-text
and predicted-duration synthesis, obtain fluent review, verify final inventories
and a standalone deployment, and document remaining limits. The toolkit is a
generic adaptation interface, not the private recipe used to produce the base.
