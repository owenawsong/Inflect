# Version-Specific Qualification

## 0.1.3 IO Repair Qualification

The only runtime change from published0.1.2 is training.atomic_json: exclusive
unique temporary files and bounded Windows replace retries for errors5/32/33.
Eight attempts use seven sleeps totalling0.81s. Other errors/OS fail immediately;
persistent locks are not bypassed. Serialization and atomic last-writer-wins
semantics remain. Model/loss/schema/profile and the other nine runtime helpers
are unchanged; checkpoint identity and determinism policies are not relaxed.

Actual private0.1.3 wheel
`f5dcafb5613686224075ac053a0b236fcfd44c1864335a8ee6095099d80c355b`
passed17 focused IO tests, a2.125s native PowerShell deny-delete stress
(25 real WinError5 replacements recovered,49 valid reads), and the separate
installed CPU suite219/1skip, including42 native and20 publication cases.
Counts are separate overlapping scopes, not a new full-suite sum.

The final long-README wheel is
`680c13a5243ba249f173a13353209e535b090d868f188b1a8d1111e14b925d9b`.
Its ten runtime files match the private0.1.3 runtime exactly. The public long
README is retained, not replaced by the short private README, and the final
wheel/embedded README have not been rebuilt after qualification.

| Final0.1.3 check | Actual scope/result |
| --- | --- |
| Installed CPU suite |236 passed,1 skipped in128.734s;42 native,20 publication and17 IO cases included |
| Publication source tests |Complete ten-helper reference fixture; all14 test files; corrected atomic test SHA75dd9996 |
| Strict CUDA smoke |One default-four-group update on local RTX3060; strict deterministic algorithms, TF32 off |
| Export and inference |CPU export plus raw-text native24k mono PCM16 WAV;121 frames/72,600 samples |
| Bounded GPU/export run |29.766s total;717,172,224B peak allocated and803,209,216B peak reserved CUDA;373,135,969B owned C output |

Environment: same qualified Windows/Python3.12.11/Torch2.6.0+cu124 dependencies,
not a fresh0.1.3 installation or Linux qualification. The one-update trainability
check covers text_encoder, duration, pitch_noise and speaker, with bridge and
generator frozen; it does not claim every selected parameter had nonzero gradient.
The input is an English mechanical alignment fixture, not evidence of new-language
quality, human preference or broad training convergence.

The first final CPU attempt recorded235 passed,1 failed,1 skipped. The failed
synthetic retry-count test called real Windows replacement after two injected
errors; one additional admitted real transient was successfully retried, then
the test's exact sleep-count assertion failed. This is retained as a failed
qualification, not waived. The corrected test captures successful replacement
bytes/destination in memory while preserving exact retry assertions. Separate
real Unicode concurrent-write/native Windows stress coverage remains. The second
attempt also recorded235 passed,1 failed,1 skipped: its moved source fixture
omitted adjacent model.py required by a source-identity test. Both failures and
their logs remain preserved. The complete fixture in the third attempt passed
the full236/1 suite without skipping or weakening either gate.
Tests are not wheel members; this correction does not rebuild/relabel wheel680c13.

No final0.1.3 resume, optimizer/WAV-parity, private A/B quality or language
improvement is asserted. The actual final GPU scope is one strict default-four-
group update, CPU export and one raw-text WAV, not the historical0.1.2 resume
sweep below. Public release still requires owner admission.

Final CPU receipt SHA256:
`fe19ec86666feb87fa5d9d4ceb869c433e8d051b9f38fe698d858362a848adea`.
Final GPU/export receipt SHA256:
`81053477cd36602f3a38f95e63eac6a13da3168bdc7f37977267973b1319f482`.
Checkpoint SHA256:
`d83709367f78bd1acd01a513b8e18df83684a7eea9b6385708c460cfd10b9df2`.
Generated raw-text WAV SHA256:
`ba3d85dc4127036ff6fba4a95cb566e98beed4140ce0b0ee52c473501b729bd9`.
Hashes identify actual artifacts, not cross-run checkpoint/WAV parity.

The actual private CPU receipt is
`toolkit-013-private-cpu-r001/run/result.json` (c094b0fa); private IO evidence is
`toolkit-013-private-io-r001/{package-receipt,native-stress-result}.json`.
None of those rewrites the failed0.1.1 CUDA result or the completed0.1.2 history
below. The separate fresh Windows CPU installation remains version0.1.1.

## Historical Version-Specific Qualification: 0.1.2

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

## Historical 0.1.2 Final-Wheel Results

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
