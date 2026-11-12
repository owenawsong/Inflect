# Current V3 Aligned-Data Workflow

This page documents the [training API](../../../finetune_v3/training.py) and
its external-alignment contract. `python -m finetune_v3` provides `vocab`, `coverage`, `smoke-dataset`,
`prepare`, `audit`, `train`, `export`, and `say`; see [actual commands](commands.md).
For verification results, installed-wheel scope, and language-quality gates,
see [tested scope](implementation-status.md). Start with the
[command recipes](commands.md) if you do not need API details.

## What You Must Supply

Start with a consenting fixed speaker, exact transcripts, legally usable clean
recordings, a reviewed language frontend, and **external aligned phone
durations** from manual annotation or an appropriate aligner. Text alone,
guessed uniform timing, and the model's own predicted timing are not accepted
substitutes for alignment labels.

Use the exact phoneme representation consumed by the model. Alignment labels
must cover that representation, not a different aligner's phone alphabet. In
particular, character-level phones, affricate spellings, stress marks, spaces,
and the two boundary tokens need a documented conversion into the native clock.
Have a reviewer inspect difficult boundaries and clusters. Passing a shape or
integer check cannot prove the labels describe the recording.

No F0 extraction, teacher feature cache, or separately supplied pitch target is
required by the current objective. The required supervision is the aligned
durations and matching waveform. This does not make timing annotation optional.

Use `vocab` and `coverage` before annotation to inspect the actual base mapping
and candidate language's unknown characters without needing alignment labels.
The separate `smoke-dataset` command generates six English examples and their
exact synthesis clocks with explicit mechanical-only acknowledgment. It is not
an aligner for arbitrary recordings and is never foreign-language proof.
Its default is prephonemized. Optional `--frontend espeak` uses fixed `en-us`
English prompts to exercise real G2P and raw-text restoration. That separate
historical 0.1.0 mechanical path passed prepare, CUDA two updates,
contract-validated export with licenses, and raw-text `say`. This does not
qualify0.1.1 CUDA training; see the [version-specific tested scope](implementation-status.md).
It is not foreign-language proof.

## Prepared Data Contract

The controller reads an already prepared directory:

```text
prepared-language/
  dataset.json
  train.jsonl
  validation.jsonl
  audio/
```

Both splits must be nonempty. Use the canonical fields below, even though the
current reader also accepts a few legacy aliases:

| Field | Required meaning |
| --- | --- |
| `audio` | Relative file path contained within the prepared directory |
| `audio_sha256` | Exact checksum of that prepared audio file |
| `text` | Exact transcript; also used for duplicate-text checks |
| `phonemes` | Full, nonempty model phoneme string, at most 510 characters |
| `phone_durations` | Positive integer frame counts for every phoneme character, **plus both boundary tokens**, in exact token order |

The new preparer additionally writes `dataset.jsonl`, `vocab.json`, and an audit
receipt, and preserves row source/consent/alignment metadata. Its output WAVs
use floating-point samples, not the v2 preparer's PCM16 conversion. Supply a
real source manifest with these additional inputs:

| Source field | Required meaning |
| --- | --- |
| `id` | Unique stable string identifier |
| `group` | Explicit leakage boundary such as a recording session |
| `source`, `consent` | Nonempty provenance/permission metadata objects; recorded user assertions, not independent rights verification |
| `phones` or `phonemes` | Explicit full phones in prephonemized mode; do not supply them as an eSpeak override |
| `alignment_phones` | Exact final phone string to which timing labels refer; already-final `phones` may serve this role in prephonemized mode |
| `alignment_source` | Object with matching `language`, nonempty `source`, and `kind` equal to `human`, `external_forced_alignment`, or `dataset_alignment`; base-predicted labels are forbidden |

Supply either the real `phone_durations` or a real contiguous timed `alignment`
covering every boundary-wrapped token with its exact token ID. The preparer can
quantize supplied timing to the native clock; it does not infer timing from
unannotated text or make uniform labels. Default preparation requires already
mono 24 kHz, on-clock audio. The CLI also exposes
[optional audited preprocessing](commands.md#optional-preprocessing):
`coverage` and `prepare` accept `--phone-mapping PATH` and `--base-vocab PATH`;
`prepare` additionally accepts `--allow-resample` and
`--clock-tolerance-samples N` (integer0..300, default0). A phone map is reviewed,
language-specific, and one-codepoint-to-one-codepoint; base-vocabulary checks
preserve existing IDs. These are not automatic ID selection or label generation.

Resampling is explicit opt-in and does not relax mono/finite/unclipped input
checks. Clock tolerance bounds adjustment to the nearest 600-sample multiple in
post-resampling 24 kHz samples. Trimming still requires a source-row
`trailing_silence_samples` declaration covering the removed tail in those same
units, and a removed tail at or below -55 dBFS. Otherwise it fails rather than
cutting speech. Padding is bounded by the admitted tolerance, changes are
recorded, and the external alignment must still match the final mapped phones
and prepared sample count. Strict defaults remain unchanged.

If `P = len(phonemes)` and `F = sum(phone_durations)`, the required invariants
are:

```text
len(phone_durations) = P + 2
each count is an integer >= 1
F is within the model and configured training frame caps
sample rate = 24000, channels = 1
waveform sample count = 600 * F
```

The first and last counts belong to the native zero-ID boundary tokens. They
are not optional padding labels. Boolean, floating, zero, negative, missing,
or wrong-length counts do not satisfy the controller's duration contract.
There is deliberately no example filled with invented duration numbers here.

At 24 kHz each duration frame is 25 ms. The reader allows a total of 2..1200
frames, then training enforces its configured cap (currently default 480 frames,
or 12 seconds). These are code bounds, not recommendations that every clip
should have those lengths. Token coverage also constrains the feasible frame
count. The controller rejects mismatched waveform length; it does not resample,
pad, truncate, or invent timing to make a row fit.

Prepare frame-aligned derivatives using an explicit reviewed conversion policy
that retains the original recording and records what changed. Do not force a
waveform to match arbitrary counts. If a manual/aligner output cannot be
represented correctly under this contract, resolve that preprocessing problem
before training.

`dataset.json` is included in the dataset hash and its `frontend` metadata enters
the run contract. The current reader is not a complete language/frontend audit:
the preparer/auditor must establish truthful language identity,
frontend resources, alignment provenance, and grouping. Do not mistake a JSON
file being readable for a qualified frontend package.

## Audit Before Calling Train

The controller first calls `data.load_prepared`, validating the full prepared
schema, index/split files, vocabulary, hashes, clocks, and connected split
relationships. A minimal hand-written directory with only three metadata files
is not a substitute for preparation. It additionally checks audio hashes,
path containment, phone/count structure,
rate/channel/sample-count agreement, and exact duplicate audio or normalized
transcripts across splits. Batch construction also rejects nonfinite or clipped
waveforms, and model vocabulary checks reject unknown phones.

Retain the stronger corpus checks from [data and validation](data-and-validation.md):
source-session grouping, overlapping crops, near duplicates, target-phone and
context coverage, transcript correctness, consent/provenance, and a separate
untouched final test. These are not all implemented by `load_rows`.

For a corpus with new phone characters, the experimental `migrate_vocab` helper
must be integrated explicitly before optimizer creation and the first training
forward. It reuses genuinely unmapped native IDs; it does not expand the model
dimensions. The controller now accepts `vocab_migration`, a local JSON file with
`phone_to_id`, a seed, and optional `copy_from`, and applies it on CPU before
device movement or optimizer creation. The map enters the run contract and is
reconstructed during export. Keep the prepared vocabulary, controller map,
checkpoint receipt, and exported frontend consistent. Low-level CPU migration
checks have a separate [model API receipt](../../../finetune_v3/MODEL_API_VERIFICATION.json);
they do not certify language pronunciation. Do not change source-package files.

## Training API and Initial Configuration

The implemented entry points are `finetune_v3.training.TrainOptions` and
`finetune_v3.training.train`. A controller uses:

```python
from finetune_v3.training import TrainOptions, train

# Integration skeleton, not a standalone command or qualified hyperparameter set.
# The caller must provide these approved paths and experiment settings.
options = TrainOptions(
    model=local_model_package,
    dataset=audited_prepared_dataset,
    output=new_run_directory,
    voice=selected_existing_voice_slot,
    device=approved_device,
    groups=("text_encoder", "duration", "pitch_noise", "speaker"),
    gradient_accumulation=1,
    steps=approved_final_update_count,
    learning_rate=approved_pilot_learning_rate,
)
# Call train(options) only after the corpus and pilot are admitted.
```

This initial configuration freezes the acoustic bridge and waveform generator;
frozen modules still propagate gradients to trainable inputs. `speaker` is an
existing six-row speaker embedding, not a new-speaker registration mechanism.
`duration` includes shared prosody context, not only a timing projection.

Every microbatch is one complete, unpadded utterance. The documented initial
workflow uses accumulation one. The controller implements bounded accumulation
for experiments that explicitly choose it, but no memory, numerical-equivalence,
or language-quality qualification is claimed for other values. `steps` counts
optimizer updates, not clips. The implementation uses AdamW with zero weight
decay and a fixed learning rate; no v2 scheduler, posterior warmup, discriminator,
AMP schedule, or decoder-unfreeze curriculum should be inferred.

Code defaults are not evidence of a good language preset. Choose and record
experiment values through the [pilot gates](pilot-and-training.md), not by
treating the existence of a default as a quality result.

## Implemented Objective

Duration supervision is smooth-L1 between `log1p` predicted duration and `log1p`
the supplied integer frame counts, including boundaries. Duration expansion is
not a substitute for this explicit timing loss.

The waveform objective averages three STFT resolutions: FFT sizes 256, 512, and
1024 with Hann windows and hop size one quarter of each FFT size. Each resolution
combines L1 log-magnitude difference with normalized magnitude error (spectral
convergence). The total objective weights duration and spectral terms according
to the recorded options. This is not a direct F0-target loss, perceptual quality
score, or guarantee of natural prosody.

Training and periodic validation reconstruct audio using the **supplied**
alignment. Validation reports a bounded prefix of held-out reconstruction losses,
not raw-text intelligibility, predicted-duration quality, or fluent listening.
Run a separate full predicted-duration synthesis panel before claiming useful
language adaptation.

## Outputs and Controller Resume

The controller writes a run contract, per-update metrics/status, checkpoints,
and a latest-checkpoint receipt. A checkpoint is paired with a
`.controller.json` sidecar binding its SHA256, step, and run-contract hash.
Resume checks the run's source identity, data/frontend/settings contract,
checkpoint membership, checksum, and model/optimizer identity.

Data order is counter-based: epoch shuffling and per-microbatch seeds are
derived from the recorded seed and optimizer/microbatch counters. There is no
active scheduler or AMP scaler to restore in this controller. This is now a
controller resume protocol, unlike the lower-level model/optimizer checkpoint
alone. Parent mechanical verification completed two updates followed by resume
through update four and a separate continuous four-update run. Small CUDA
nondeterminism was observed: the passes do not establish bit-exact equivalence
or language learning. See [evidence sources](implementation-status.md#cli-integration-checks-before-promoting-examples).

Preserve checkpoint and matching sidecar together. Keep data, frontend,
voice/groups, loss, learning rate, accumulation, and relevant settings fixed.
Use a new run for changed conditions rather than bypassing a contract rejection.
The final update target may be extended under the controller's existing
contract; this is not permission to silently change the experiment objective.

CLI export also validates the checkpoint/controller sidecar SHA256 and
run-contract hash before loading checkpoint weights, and checks the restored
step afterward. Keeping only a model-state file loses the controller binding
needed for this route.

## Export and Say Handoff

The lower-level `export_model` produces a strictly loaded model core, not a
complete raw-text language package. CLI `export` now adds a hashed
`frontend.json` with explicit language/frontend identity, vocabulary, and
`target_voice`. CLI `say` defaults to that target and rejects any different
explicit voice as unqualified. Six architecture slots do not establish
six-voice adaptation quality.

Full package assembly still needs matching frontend resources/configuration,
checksums, vocabulary migration receipts if used, and rebuilt outer manifests.
CLI export requires the full source package's root license documents and core
notices, retains original license/notice texts under `source-notices/`, hashes
them, and recomputes the returned final manifest checksum. Retained historical
paths/hashes describe source provenance, not adapted weights; upstream licenses
are not rewritten. See [notices.py](../../../finetune_v3/notices.py).
Required retained source texts include `core/LICENSES/Apache-2.0.txt` and
`core/LICENSES/StyleTTS2-MIT.txt`.
Missing or mismatched language resources must fail clearly; there must be
**no silent English fallback**. A prephonemized export refuses raw-text `say`
rather than using the original English frontend.

For eSpeak raw-text inference, the actual engine version must equal the saved
frontend version; a missing or mismatched version raises an error. The optional
English eSpeak fixture now has a parent-verified end-to-end mechanical pass.
Engine `(1, 52, 0)` matched, default target Cassidy worked, and raw text
`A small model can still speak clearly.` yielded mono24 output with 120 native
frames (3 seconds). This adds no language/perceptual-quality or independent
wheel-install claim. See [receipts](implementation-status.md#cli-integration-checks-before-promoting-examples).

Test `say` from that exported artifact without the training checkout or source
recordings. Distinguish explicit-phoneme inference from true raw-text inference,
and verify which was used in every evaluation. Keep the original public package
unchanged. A completed training/export status is still not language-quality or
human approval.
