# Current CLI: Coverage, External Alignment, and Mechanical Smoke

Follow these recipes from vocabulary inspection to externally aligned data,
pilot training, and export. Use the separate English smoke to check the tool
chain before working on a language corpus. See
[tested scope and remaining gates](implementation-status.md) for the verified
software/installation results and the separate language-quality requirements.

Install from the repository root using Python3.12 and an appropriate PyTorch
environment:

```sh
python -m pip install './finetune_v3'
python -m finetune_v3 --help
```

The installed `inflect-v3-adapt` console entry point uses the same parser.
See the [tested dependency snapshot](../../../finetune_v3/requirements-tested.txt)
for the existing verification environment; choose the CPU/CUDA Torch build
separately. The snapshot does not certify a fresh complete environment.
The original Apache `LICENSE` is included with legacy-compatible setuptools
license-file metadata. Version 0.1.0 wheel build r004 passed raw-text generation
from an unrelated Temp CWD with an isolated --target installation and existing
qualified dependencies, not a freshly downloaded complete environment. That
0.1.0/r004 result remains historical. The current 0.1.1 full CPU suite reported
199 passed, 1 skipped, with 42 real-model cases included in those passes.
It also passed offline isolated-target
installation, strict export and raw-text inference from an unrelated Temp CWD.
Its 120-frame mono24 PCM16 output is a mechanical check with existing qualified
dependencies, not a fresh environment or new-language quality result. See the
[version-specific test scope](implementation-status.md#staged-011-coverage).
First obtain and verify the [complete pinned base](pinned-base.md). If coverage needs additions, construct the [coordinated vocabulary/map/migration](coordinated-vocabulary.md). Choose the separate [prephonemized or eSpeak end-to-end branch](two-input-workflows.md) before alignment; only the latter produces a raw-text export.

Replace `./inflect-mini-v3` with the verified local Mini package if it is elsewhere.
Assume `./work/adaptation` already exists. New output files/directories must not
exist, their parents must exist, and they must be outside the source package.
The installation step installs declared dependencies. The model commands below
do not download a base or align arbitrary raw recordings automatically. Keep
the full source package, including its root license documents and core notices:
CLI export now refuses a bare model core without that provenance.

## 1. Extract the Verified Base Vocabulary

```sh
python -B -m finetune_v3 vocab --model ./inflect-mini-v3 --output ./work/adaptation/base-vocab.json
```

This verifies the local model-core inventory, writes the actual character-to-ID
map, and reports genuinely unmapped IDs. It does not select new phone IDs or
initialize model rows. Retain its source identity with the experiment records.

## 2. Audit Frontend Coverage Before Audio Alignment

For eSpeak mode, provide a UTF-8 JSONL prompt manifest with `text` and optional
`id`. It needs no waveform or timing labels. For example, these are authored
prompts, not supplied recordings:

```jsonl
{"id":"coverage-01","text":"Proszę zaczekać."}
{"id":"coverage-02","text":"Łukasz jedzie do Łodzi."}
```

```sh
python -B -m finetune_v3 coverage --manifest ./data/pl-prompts.jsonl --vocab ./work/adaptation/base-vocab.json --language pl --frontend espeak --output ./work/adaptation/pl-coverage.json
```

Read `status`, `unknown_codepoints`, and the encoded examples in the report.
Successful command execution is not necessarily coverage success: unknowns are
reported as `explicit-vocabulary-migration-required`. The audit does not generate
alignment labels or certify pronunciation. Missing language resources, switched
language flags, and malformed input can still produce errors.

For an externally reviewed G2P, use `--frontend prephonemized` and supply `phones`
or `phonemes` on every prompt row. Compare the full final representation against
the vocabulary before collecting/aligning a large corpus. Do not remove
unknowns to make the audit pass. If additions are required, review a separate
unused-ID migration and the matching prepared vocabulary; coverage itself does
not edit either the vocabulary or model.

Both `coverage` and `prepare` accept `--phone-mapping PATH` and
`--base-vocab PATH`. Use an already reviewed language-specific phone map
(`language`, `name`, and one-codepoint-to-one-codepoint `replacements`) to make
explicit representation changes; deletions and expansions are not allowed.
`--base-vocab` checks that a reviewed extended vocabulary preserves original
mapped IDs and assigns additions only to previously unused IDs. Neither flag
chooses IDs or creates duration labels; model-row initialization still uses the
separate explicit migration at training time. Use the same map and vocabulary
contract during coverage and preparation. For example:

```sh
python -B -m finetune_v3 coverage --manifest ./data/pl-prompts.jsonl --vocab ./work/adaptation/reviewed-vocab.json --base-vocab ./work/adaptation/base-vocab.json --phone-mapping ./work/adaptation/pl-phone-map.json --language pl --frontend espeak --output ./work/adaptation/pl-mapped-coverage.json
```

## 3. Prepare Real Recordings With External Alignment

First supply a genuine aligned source manifest as specified in
[the workflow contract](workflow.md#prepared-data-contract). It needs real
boundary-inclusive durations or externally timed alignment, final aligned
phones, explicit source/consent/group metadata, and suitable audio. The tool is
not a forced aligner. Do not use guessed uniform counts or English base-model
predictions as labels for a separate Polish recording.

Assume `./data/pl-aligned.jsonl` contains reviewed external labels and
`./data/pl-audio` contains its recordings. `reviewed-vocab.json` means the base
map unchanged if no additions are needed, or the reviewed extension preserving
all existing IDs when a migration is required.

```sh
python -B -m finetune_v3 prepare --manifest ./data/pl-aligned.jsonl --audio-root ./data/pl-audio --output ./work/adaptation/pl-prepared --vocab ./work/adaptation/reviewed-vocab.json --language pl --frontend prephonemized --split-seed 2026 --validation-fraction 0.1
python -B -m finetune_v3 audit --dataset ./work/adaptation/pl-prepared/dataset.json
```

The split settings above are explicit API example values, not a qualified data
allocation. Design independent source groups and an untouched final test before
preparation. Default CLI preparation expects mono24 audio already on the
600-sample frame clock; it does not silently repair timing or invent labels.

If the recording's labels were established against the exact configured eSpeak
output, `--frontend espeak` is another supported preparation mode. Use
`alignment_phones` to bind labels to its exact final output; do not pass caller
`phones` as an eSpeak override. Changing G2P output after alignment invalidates
that correspondence and needs reviewed re-preparation.

### Optional Preprocessing

The CLI exposes audited conversion without requiring a Python wrapper:

| Option | Behavior |
| --- | --- |
| `--phone-mapping PATH`, `--base-vocab PATH` | The same reviewed representation/base-ID checks used by coverage |
| `--allow-resample` | Explicitly permit conversion to 24 kHz; input must still be mono, finite, and below full scale |
| `--clock-tolerance-samples N` | Permit adjustment to the nearest 600-sample clock within N post-resampling samples; integer0..300, default0 |

Strict defaults are unchanged: no resampling opt-in and zero clock tolerance.
Resampling does not itself permit off-clock audio. When an admitted adjustment
needs trimming, the source row must declare reviewed `trailing_silence_samples`
in **24 kHz sample units**, covering the removed samples; the removed tail must
also be at or below -55 dBFS. Non-silent tails are rejected. Padding is limited
to the admitted tolerance. No gain repair, automatic downmix, or word cutting
is introduced, and conversion/clock changes are recorded in prepared metadata.

For already aligned recordings needing explicitly admitted resampling and
clock adjustment, this example permits the maximum bound (300 samples =
12.5 ms); choose a smaller reviewed bound where appropriate:

```sh
python -B -m finetune_v3 prepare --manifest ./data/pl-aligned.jsonl --audio-root ./data/pl-audio --output ./work/adaptation/pl-prepared-resampled --vocab ./work/adaptation/reviewed-vocab.json --base-vocab ./work/adaptation/base-vocab.json --phone-mapping ./work/adaptation/pl-phone-map.json --language pl --frontend prephonemized --allow-resample --clock-tolerance-samples 300
```

The supplied alignment must still bind to the **final mapped phones** and
exact prepared clock. These options do not generate phones automatically in
prephonemized mode, infer alignment, or turn guessed uniform counts into labels.
Run `audit` on the resulting dataset before training.

## 4. Train an Approved Real-Data Pilot

The following PowerShell command uses caller-chosen `$PilotSteps`,
`$PilotLearningRate`, and `$MaxFrames`. Set them only from the approved pilot
design; this guide supplies no qualified foreign-language values.

```powershell
python -B -m finetune_v3 train --model ./inflect-mini-v3 --dataset ./work/adaptation/pl-prepared/dataset.json --output ./work/adaptation/pl-pilot --voice cassidy --device cpu --groups text_encoder duration pitch_noise speaker --gradient-accumulation 1 --steps $PilotSteps --learning-rate $PilotLearningRate --max-frames $MaxFrames
```

If new phone rows were explicitly admitted, add
`--vocab-migration ./work/adaptation/pl-migration.json`. That file records
`phone_to_id`, the initialization seed, and any explicit `copy_from`; it must
produce exactly the vocabulary in the prepared data. Do not edit the official
source model or reassign mapped IDs. Keep the same migration during resume and
export. Select another existing voice slot only as a separately recorded target.

The initial groups freeze bridge/generator weights. `duration` also includes
shared prosody context. No F0 extraction is required. The supervised objective
uses external duration labels and waveform spectral reconstruction, not
automatic foreign-language alignment. Evaluate predicted-duration inference on
held-out text separately from aligned reconstruction loss.

## 5. Separate Mechanical Smoke: Not a Language Experiment

This branch is deliberately different from real-data preparation:

```sh
python -B -m finetune_v3 smoke-dataset --model ./inflect-mini-v3 --output ./work/adaptation/mechanical --acknowledge-mechanical-only
python -B -m finetune_v3 audit --dataset ./work/adaptation/mechanical/prepared/dataset.json
```

`smoke-dataset` creates six English Cassidy examples locally from the base model,
records the exact durations used to generate those same waveforms, then invokes
preparation. It writes `SMOKE-SCOPE.json`. This is a mechanical
preparation/training/resume/export fixture, not human recordings, a new language,
or evidence that the tool can align unrelated raw audio. Its generated clocks
must never be transplanted onto external recordings.

The default frontend is `prephonemized`. The optional eSpeak variant is fixed
to the same `en-us` mechanical prompts, not a selectable foreign language:

```sh
python -B -m finetune_v3 smoke-dataset --model ./inflect-mini-v3 --output ./work/adaptation/mechanical-espeak --frontend espeak --acknowledge-mechanical-only
```

It obtains the prompt phones from eSpeak, generates their corresponding English
waveforms and exact clocks, then prepares an eSpeak-identified dataset. This
supports a separate train/export/raw-text-restore test using new output names.
**Historical 0.1.0 English eSpeak mechanical verification passed**:
six self-generated examples, prepare, CUDA two updates, contract-validated
export with licenses, then raw-text inference. This is separate evidence from
the earlier phoneme-input pass, not permission to use a prephonemized export as
a raw-text model. From the separate eSpeak run/export, the exercised input is:

```sh
python -B -m finetune_v3 say --model ./work/adaptation/mechanical-espeak-export --text "A small model can still speak clearly." --output ./work/adaptation/espeak-export-rawtext.wav
```

The saved and actual engine versions matched `(1, 52, 0)`; default target
Cassidy worked. WAV metadata was mono24, 72,000 samples, 120 native frames
(3 seconds). See [receipts](implementation-status.md#cli-integration-checks-before-promoting-examples).
This verifies English mechanical restoration, not foreign-language alignment,
linguistic quality, or a clean wheel installation.

The parent's historical 0.1.0 CUDA smoke used the following settings. These tiny counts
and learning rate are **mechanical test settings only**, not an adaptation
recommendation. Running them requires explicit local CUDA admission.

```sh
python -B -m finetune_v3 train --model ./inflect-mini-v3 --dataset ./work/adaptation/mechanical/prepared/dataset.json --output ./work/adaptation/mechanical-run --voice cassidy --device cuda --groups text_encoder duration pitch_noise speaker --steps 2 --learning-rate 0.00001 --gradient-accumulation 1 --checkpoint-every 2 --validation-every 2 --max-validation-clips 1 --max-frames 120 --seed 2026 --threads 2
python -B -m finetune_v3 train --model ./inflect-mini-v3 --dataset ./work/adaptation/mechanical/prepared/dataset.json --output ./work/adaptation/mechanical-run --voice cassidy --device cuda --groups text_encoder duration pitch_noise speaker --steps 4 --learning-rate 0.00001 --gradient-accumulation 1 --checkpoint-every 2 --validation-every 2 --max-validation-clips 1 --max-frames 120 --seed 2026 --threads 2 --resume ./work/adaptation/mechanical-run/checkpoints/step-0000002.pt
python -B -m finetune_v3 export --run ./work/adaptation/mechanical-run --checkpoint ./work/adaptation/mechanical-run/checkpoints/step-0000004.pt --output ./work/adaptation/mechanical-export
python -B -m finetune_v3 say --model ./work/adaptation/mechanical-export --phonemes "həlˈO." --output ./work/adaptation/mechanical-hello.wav
```

The step-2 checkpoint and matching `.controller.json` sidecar belong together.
Step 4 is the final update target, not four additional updates. The local record
shows resumed steps 1..4 and a separate continuous four-update run completed
with acoustics frozen and `language_quality_approved=false`. Small CUDA
nondeterminism was observed; the comparison is **not bit-exact**. These passes
do not establish held-out perceptual or foreign-language quality.

Do not overwrite an earlier run or export to repeat the smoke. Use new output
names and preserve receipts. A CPU run is a distinct test using `--device cpu`
consistently; unavailable CUDA is an error, not a silent fallback.

## 6. Export the Target Voice and Correct Frontend

For a real-data pilot, use the same `export --run ... --checkpoint ... --output
...` form with its selected checkpoint. Export adds a hashed `frontend.json`
containing language/frontend identity, vocabulary and `target_voice`.

Before loading model weights, export verifies checkpoint membership, its
`.controller.json` sidecar, checkpoint SHA256, and the run-contract hash, then
checks the restored step agrees. A checkpoint without its matching controller
receipt/contract is not an export-ready substitute.

`say` defaults to that target. An explicit different `--voice` is refused as
unqualified. The existence of six architectural slots is not proof that all
six retain language quality after one-slot adaptation. The low-level model API
can expose more slots; that is not permission to bypass the adapted CLI guard.

Prephonemized exports require `--phonemes`; `--text` must fail rather than use
English G2P. For an export actually prepared with the eSpeak frontend, the
raw-text form is:

```sh
python -B -m finetune_v3 say --model ./work/adaptation/pl-espeak-export --text "Proszę zaczekać." --output ./work/adaptation/pl-raw-text-check.wav
```

This shows the supported interface, **not a passed Polish raw-text test**.
The matching language resources must be installed and correctly identified;
the eSpeak engine version must match the saved training frontend version.
Unknown phones, frontend failures, or language switches must not trigger an
English fallback. The English mechanical fixture cannot qualify this path.

CLI export now requires the original full source package: `LICENSE.md`,
`LICENSING.md`, root/core notices, `core/LICENSE.md`, `core/LICENSE_SCOPE.json`,
and `core/LICENSES/` texts, explicitly including `Apache-2.0.txt` and
`StyleTTS2-MIT.txt`. It retains original texts under `source-notices/`,
copies the governing root license documents, hashes retained files in the final
manifest, and updates its returned manifest checksum. Historical source paths
and weight hashes remain provenance, not adapted-weight claims. No third-party
material is relicensed.

Strict model-core readback, retained notices, and successful WAV writing still
do not establish a new language's quality or every deployment environment.
Before sharing your artifact, follow the [release gates](implementation-status.md#remaining-release-gates):
verify the final installation/inventories and evaluate held-out speech with
fluent reviewers.
