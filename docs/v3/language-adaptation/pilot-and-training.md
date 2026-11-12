# From Preparation to a Qualified Adaptation

Use this sequence to move from mechanically working tools to a useful language
pilot and a verified artifact. The [command recipes](commands.md) cover execution;
the [workflow](workflow.md) explains the supervised objective and data contract.
Choose pilot settings from measured corpus behavior, not from the tiny English
smoke. Full-corpus adaptation uses the larger approved corpus; it does not
recreate the original model from scratch.

## What Is Implemented So Far

The [v3 model bridge](../../../finetune_v3/model.py) loads a local Mini
package, verifies its model-core inventory, preserves fixed conditioning, and
exposes named trainable groups. The new
[training.py](../../../finetune_v3/training.py) adds `TrainOptions`/`train`,
strict aligned-row consumption, supervised duration and multi-resolution
spectral loss, optimization, and a counter-based resume protocol. Preparation,
frontend, and CLI files are also implemented. It is not the old VITS trainer.
This is source inspection, not a completed end-to-end language training test.

Its current contract has important consequences:

- Batch size is exactly one full utterance. Unequal padded utterances are
  rejected; do not copy v2 batch-size presets into this bridge.
- The supplied alignment is hard, monotonic, covers every token including
  boundaries, and uses 600 waveform samples per frame at 24 kHz (25 ms).
- Duration supervision must be explicit. The bridge does not endorse treating
  its own predicted durations as ground-truth timing for new recordings.
- Model dimensions and fixed voice conditioning remain unchanged. Experimental
  pre-training migration can assign new phones to actually unused native IDs;
  it does not resize token tables or add a runtime language selector.
- Groups include `text_encoder`, `duration`, `pitch_noise`, `speaker`, `bridge`,
  and `generator`. The `duration` group includes shared pitch context, so it is
  not an acoustically isolated duration-only control.
- Harmonic excitation is not differentiable through its source path. Pitch can
  still affect differentiable bridge inputs; verify the actual gradient path
  for any proposed pitch loss rather than assuming waveform loss supervises all
  components equally.

The controller's initial selection is `text_encoder`, `duration`, `pitch_noise`,
and `speaker`, with bridge/generator frozen and accumulation one. It requires
no F0 extraction. This is the implemented starting workflow, not a qualified
language preset. Alignment provenance, gradient checks, resume tests, and
held-out output still need evidence. See [workflow](workflow.md) and
[implementation status](implementation-status.md).

## Gate 1: Reproduce the Unchanged Base

Before changing weights, load the exact pinned base and compare the bridge's
evaluation output with the independent public runtime on a fixed phoneme/voice
panel. Use the declared seed and output-rate policy. Record exact equality or
the justified numerical tolerance, not just that both WAVs play.

Confirm that the proposed training parameter groups cover exactly what is
intended, that all other parameters stay frozen, and that frozen modules still
pass gradients to trainable upstream inputs when needed. Do not put a frozen
decoder in `no_grad` if that severs the learning signal required by the objective.

## Gate 2: Data and Training-Mechanics Smoke

Use a small, deliberately varied subset with independent training and
development groups. Include supported clip lengths, rare target sounds,
boundaries, and several transcript patterns. A mechanics smoke establishes
that a batch can be read, aligned, optimized, saved, resumed, exported, and
loaded; it does not establish new-language quality.

Record input and target dimensions, token/frame clocks, finite losses and
gradients, gradient norms, changed and unchanged parameter groups, optimizer
updates, example exposure, seed policy, and hardware/precision. Save at an
optimizer boundary and verify same-run resume with the identical data/frontend
and configuration. Never bypass identity checks to reuse a checkpoint.

If alignment is externally supplied, retain its provenance and conversion
settings. Check sample rates, boundary treatment, and whether every token has
valid duration. Supervised-alignment reconstruction and predicted-alignment
inference must be tested separately; successful reconstruction can conceal a
broken duration predictor.

## Gate 3: A Representative Language Pilot

Choose the pilot subset by coverage and independent sources, not just the
shortest or easiest files. Establish the smallest approved configuration that
can answer the current question, and change one major factor at a time. Start
from a clean base for comparisons unless continuation is part of the design.

Use an agreed held-out panel throughout the pilot. Keep a separate final test
untouched. Assess raw-text frontend behavior, full predicted-duration synthesis,
pronunciation, clipping, silence, truncation, voice stability, and native-rate
artifacts. Sample checkpoints at declared intervals; do not select the final
checkpoint solely by training loss or one attractive sample.

No fixed step budget is a language-learning guarantee. Measure updates per
second, actual audio exposure, coverage, and development trends. Stop and repair
the data or objective if output develops severe instability, unintelligible
phones, persistent omitted endings, or identity collapse. Record failed pilots
alongside successful ones instead of silently changing the selection set.

## Memory Planning Without Hardware Promises

An inference weight file is not a training-memory estimate. Budget separately
for resident frozen and trainable weights, trainable gradients, optimizer
states, saved activations, temporary kernels/workspaces, decoded waveforms,
features, validation, and checkpoint serialization. A discriminator or external
alignment/feature network adds its own cost if the final trainer uses one.

For rough accounting only, FP32 parameters and gradients use about 4 bytes per
element each; two FP32 Adam moment buffers add about 8 more bytes per trained
element. This excludes activations, temporary buffers, possible master copies,
frozen modules, and runtime overhead. It is not a VRAM requirement or a
guarantee that a named GPU will fit.

Full-utterance text and alignment processing can remain expensive even when a
waveform loss uses short segments. The v2 collator pads to the longest utterance;
the current v3 bridge forbids that padding and processes one full utterance.
Its length distribution still matters. Profile the longest admitted clips as
well as typical clips and the first optimizer step, when state is allocated.

The controller supports accumulation of batch-one utterances without padded
multi-utterance batches; the initial workflow uses accumulation one. Nominal
effective batch is microbatch size times accumulation count times worker count
if distributed training is actually supported (not implemented by the inspected
controller). This does not guarantee identical optimization to a simultaneous
larger batch. Keep loss scaling, clipping, and optimizer updates consistent.
The controller does not implement AMP; the following reference explains scaling
requirements if AMP is added, not an available v3 option.
[PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

Measure peak allocated and reserved device memory, process RAM, data-loader
workers, throughput, and validation peaks in the chosen environment. PyTorch's
cached reserved memory differs from live tensor allocation; `empty_cache()`
does not free live tensors. Fix workload size and diagnose memory before
copying allocator settings from another run.
[PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management).

If memory is exhausted, first identify whether the peak is model state,
activations, waveform processing, validation, or host prefetching. Reduce the
appropriate workload, or narrow trainable groups only if it remains a valid
experiment. Split long source utterances at reviewed linguistic boundaries;
do not truncate model inputs or silently drop words. Precision changes,
activation checkpointing, segmentation, and accumulation need explicit trainer
support and fresh numerical tests. Never assume v2 `micro-12gb` or `nano-8gb`
preset names qualify v3 Mini.

## Host RAM, Disk, and Time

One hour of 24 kHz mono PCM16 audio is approximately 172.8 MB of sample payload
(about 164.8 MiB), excluding containers and metadata. A fully decoded FP32 copy
is approximately twice that. This is arithmetic, not a disk-allocation
recommendation: retain room for originals, prepared derivatives, features,
checkpoints, temporary copies, and evaluation audio. Feature caches can be much
larger than the waveform payload.

Workers and prefetch queues can multiply resident decoded data. Begin memory
measurement with controlled worker counts and record the actual loader design.
Do not load a whole corpus into RAM by assumption. Keep the approved storage
and retention policy separate from these model-training decisions.

Estimate elapsed time from measured steady-state optimizer updates plus data
preparation, validation, checkpoints, and export. Re-measure after unfreezing
more groups, changing clip lengths, or changing precision. Do not extrapolate
short event-control fits or inference real-time factors to acoustic-model
language adaptation.

## Gate 4: Full-Corpus Adaptation

Proceed only after the pilot gives useful held-out evidence and the resource
budget is measured. Freeze the approved corpus, frontend, alignment method,
base revision, objective, trainable groups, seed policy, optimizer settings,
selection rule, and checkpoint/validation schedule. Record any departure as a
new experiment rather than changing the meaning of an existing receipt.

Use validation trends and fluent-speaker review for checkpoint selection, then
run the untouched test once the choice is fixed. Repeated tuning against that
test turns it into development data. If retained English or multiple voices
are claimed, use a separate retention panel and report regressions. Training
one speaker slot can still change shared weights used by other slots.

## Gate 5: Export and Independent Inference

Export to a separate adapted package. Preserve original weights and inventory;
never overwrite or disguise the official package as a different language.
Verify that only intended trainable tensors changed, fixed conditioning stayed
fixed where promised, and the exported loader rejects incompatible schemas.

Test from the adapted artifact in a clean environment without the training
checkout, alignment targets, or source recordings. Generation should use text
or declared full phonemes and learned weights, not source-audio retrieval.
Package exact frontend resources and their licenses. Test every published
runtime/precision separately; v2 ONNX support does not establish v3 ONNX support.

The release report should include data/split quantities and provenance, frontend
and model identities, training configuration, failures, per-item held-out
results, fluent-speaker review status, known limits, and tested deployment
paths. Human review remains pending until actual human review occurs.

The adapted CLI export records one `target_voice` and refuses other slots as
unqualified. Do not bypass that guard to turn a one-target pilot into a six-voice
claim. Mechanical English smoke and successful CUDA resume/export do not lift
the language-quality or publication gates.
