# Version-Specific Qualification: 0.1.2

The final publication wheel is
`70d94a784ef718680f9dd632e5fa3ba488c6212ccd22f7d2d213e1cdc5606ff8`,
48,810 bytes. Completed checks below identify this artifact. The embedded long
README SHA is
`c4b5d86edb1795f5764a8c9c1a737d9b40c9e075266d6e3c3438a56092c43b59`
and remains unchanged. Its conservative four-group caution predates the final
checks; this linked external scope records their completed outcome without
changing wheel metadata.

## Implementation and Resume Boundary

Only model.py and training.py changed Python implementation relative to0.1.1:
the parameterless exact-path training pad and explicitly reflected centered
STFT inputs. Other eight Python helpers are byte-preserved. All ten runtime
helpers match the qualified private0.1.2 implementation. Public inference code,
source model weights, frontend/labels, loss objective and serialized inference
schema are unchanged.

New contracts bind `native24-reflect-cat-v1`. Missing, older or altered profiles
reject before model/checkpoint loading; same-profile resume also needs the full
original contract and checkpoint/controller sidecar. No tolerance, silent
conversion or cross-version optimizer continuation is provided. Valid legacy
checkpoints may still export under their original hashes, contracts and notices.
Exported weights may initialize a new run; that is not optimizer resume.

## Actual Final-Wheel Results

| Check | Scope/result |
| --- | --- |
| CPU suite |219 passed,1 skipped;42 native and20 new publication cases included in219 |
| Publication-test portability |Actual publication snapshot; private reference-root variable absent |
| Strict CUDA SMOKE |Default four groups, one update,20.641s |
| Strict CUDA MAIN |Eight updates,71.032s: train2, resume-to4, continuous4 |
| Resume equality |Exact full checkpoint payload and optimizer-state equality |
| CPU export/raw-text |Both paths passed from unrelated working directory; byte-identical WAVs |

Environment: Python3.12.11, Torch2.6.0+cu124/CUDA12.4, RTX3060, deterministic
algorithms enabled, TF32 disabled; existing qualified dependencies. Six existing
English Cassidy mechanical exact-clock clips; groups
`text_encoder duration pitch_noise speaker`, max80frames, seed2026, threads2.
Trainability admission verifies selected groups, **not** nonzero gradients for
every selected parameter. The bridge/generator remained frozen.

Final WAV SHA:
`95c39d1bf1e0ddaed8a8978c2d874029974a23d3b5d48f81951c4f4a70c87ae4`.

Maintainer evidence IDs:
- `toolkit-012-package-r003/PACKAGE-RECEIPT.json`: final build/install and README/runtime-byte identity.
- `toolkit-012-package-r003/build-inputs.json`: source/publication-test pins.
- `toolkit-012-final-cpu-r001/run/result.json`: actual219/1/42/20 and final wheel hash.
- `toolkit-012-final-default-r001/smoke/result.json`: strict default-group one-update pass.
- `toolkit-012-final-default-r001/main/result.json` and `main/checkpoint-comparison.json`: exact full payload and WAV equality.

## Preserved Earlier Evidence

Private0.1.2 wheel2d402b separately passed speaker-only SMOKE1/MAIN8
(19.594s/62.016s), CPU219/1skip/42-native, exact checkpoint/optimizer and CPU
export/WAV equality. It is not the final metadata wheel; counts are not summed.

The strict0.1.1 CUDA backward failure remains a failure; determinism was not
relaxed. Its separate fresh Windows CPU qualification used wheel07c917b,
CPU Torch2.6.0+cpu and eSpeak1.52.0 and passed after two preserved
setup-qualification failures. It is not fresh-dependency evidence for0.1.2.

No new-language or Polish quality, human listening approval, ASR/MOS gain,
unseen-data success or arbitrary platform/hyperparameter support follows.
Old-versus-new floating-point training trajectory equality is not asserted.
