# Mini v3 Model / Export Handoff

This is the current model-bridge API. It supersedes the older README paragraph
that said vocabulary migration was unsupported. The parent owns data preparation,
frontend coverage, CLI, trainer/controller, GPU admission, and full-package assembly.

## Load And Configure

```python
from finetune_v3 import load_mini, save_checkpoint, load_checkpoint, export_model

model = load_mini(local_package_or_model_core)
receipt = model.migrate_vocab(phone_to_unused_id, seed=42, copy_from=optional_old_phones)
model.configure_trainable(("text_encoder", "duration", "pitch_noise", "speaker"))
model = model.to(device)
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], ...)
model.train()
out = model.forward_grad(ids, lengths, aln, sid, seed=explicit_microbatch_seed)
```

Omit `migrate_vocab` when no additions are required. It must run on CPU before
the optimizer, any training forward, or checkpoint restore. Native178 remains
unchanged: the public vocabulary maps 114 characters, leaving 63 unmapped IDs.
Pass new single-character phones and distinct IDs currently absent from the map.
All original mappings and all embedding rows except the explicit new IDs remain
bit-identical. Only new rows in the ALBERT token embedding `[178,128]` and text
encoder embedding `[178,384]` are initialized. Defaults are deterministic normal
initialization; `copy_from={new_phone: existing_phone}` explicitly copies that
old phone's row in each embedding. No linguistic relation is inferred automatically.

The returned receipt binds base/new contracts, source hashes, initializer seed,
mapping, and retained/new row hashes. The model records it in adaptation
checkpoints and export receipts. Frontend phoneme coverage is not certified here.

## Aligned Forward

All inputs must share the model device. Keep full utterances; batch size is one,
without padding. The released acoustic normalization is not padding-invariant.

| Field | Shape / meaning |
| --- | --- |
| `ids` | int64 `[1,T]`, zero boundaries, T=3..512; inner IDs must be in the explicit vocabulary |
| `lengths` | int64 `[1]` containing exactly T |
| `sid` | int64 `[1]`, Hope/Sophia/Cassidy/James/Jasper/Mark = 0..5 |
| `aln` | FP32 `[1,T,F]`, hard monotonic one-hot, every token covered, F=2..released cap |
| `dur` | `[1,T]` prediction, differentiable; supervise separately against actual duration labels |
| `f0`, `noise` | `[1,2F]`, differentiable through the acoustic bridge |
| `asr`, `asr_tok` | `[1,512,F]`, `[1,512,T]`; acoustic features, not transcripts |
| `hidden`, `audio` | `[1,512,2F]`, `[1,600F]`; native mono 24 kHz |

Integer `phone_durations` include both zero-boundary tokens, each at least one
frame. Parent builds `aln` from these counts. The bridge does not round, rescale,
stretch, crop, or invent these labels. Audio must contain exactly
`600 * sum(phone_durations)` samples. One frame is 25 ms; the F0/noise clock is
12.5 ms. For fractional aligner output, the parent must specify and validate its
own conversion policy rather than silently rounding labels.

Public **prediction** uses `round(dur / speed).clamp(min=1)` per token (Torch
ties-to-even), then hard expansion. That operation is not differentiable and is
not a supervised-label generator. F0 also cannot backpropagate through the public
generator's harmonic excitation/STFT branch, which is under `no_grad`; its path
through the bridge remains differentiable. Frozen bridge/generator parameters do
not block input gradients. `decode=False` omits hidden/audio for feature-only work.

`seed` fixes source noise and training dropout and restores caller Torch RNG
state. The parent chooses its explicit per-example/per-step seed policy. CPU
parity is verified; CUDA training/parity is a separate parent smoke gate.
The pinned generator uses `TorchSTFT`, whose window is an ordinary CPU tensor
explicitly moved to each transform/inverse input device by the public method.
The alternate `CustomSTFT` is not selected. No STFT runtime patch is introduced.

## Save, Resume, Export

```python
save_checkpoint(model, new_path, step=step, optimizer=optimizer)

# Resume from the SAME original package, same groups, and SAME migration:
resumed = load_mini(original_package)
resumed.migrate_vocab(same_mapping, seed=same_seed, copy_from=same_old_phones)
resumed.configure_trainable(same_groups)
resumed = resumed.to(device)
resumed_optimizer = AdamW([p for p in resumed.parameters() if p.requires_grad], ...)
receipt = load_checkpoint(resumed, checkpoint_path, optimizer=resumed_optimizer)

model.eval()
export_receipt = export_model(model, new_model_core_directory)
```

Checkpoint format is `inflect-mini-v3-adaptation-v1`, loaded with
`weights_only=True`. Source identity, factory/vocabulary/conditioning contract,
migration receipts, group selection, tensor keys/shapes/dtypes and optimizer
parameter names/order are checked. Adam/SGD moment shapes and finite values are
validated before live model mutation. Loading locks further vocabulary migration.
Parent controller/scheduler/scaler/data position and RNG resume are not included.

Exports copy public runtime code unchanged and strictly read back all tensors
through its public loader. This is a model-core export, not a complete raw-text
package. The parent must retain licenses/notices/frontend and rebuild outer
inventories. If the parent adds files or changes the model-core manifest afterward,
it must recompute `manifest_sha256` and revalidate the final inventory.
No existing destination or source weights are overwritten. Compatibility tests
and self-generated English fixtures are mechanical checks, not foreign-language
supervision, perceptual approval or a quality benchmark.
