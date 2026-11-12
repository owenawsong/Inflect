# Tested Scope and Release Gates

The tools have mechanical verification on the native24 Mini architecture.
These results establish software behavior, not Polish, multilingual, or
perceptual quality. This page summarizes parent verification as of 2026-11-12.

## Staged 0.1.1 Coverage

The 0.1.1 fixes verified-source importing and exact post-map alignment binding
have completed CPU model-inclusive, export/raw-text and installed-wheel checks.
The heading is retained for existing links; these are actual completed checks,
not pending integration. No new language data or language-quality claim is
part of this update. See the [changelog](../../../finetune_v3/CHANGELOG.md) and
[security notes](security-notes.md).

| Check | Actual result and boundary |
| --- | --- |
| Full source suite | 199 passed, 1 skipped, 23 warnings in 139.15s pytest time; monitored child phase 141.44s |
| Real-model cases | All 42 executed, including the new native combining-map migration/checkpoint/export regression; included in 199, not an additional total |
| Earlier targeted and lightweight checks | 66 passed/1 skipped and 157 passed/1 skipped; overlapping component scopes, not counts to add to 199 |
| Existing environment | Python 3.12.11, Torch 2.6.0+cu124 executed on CPU, CUDA hidden; existing qualified dependencies, not a fresh dependency install |
| Installed CLI export | Existing English eSpeak mechanical checkpoint exported on CPU through controller/checkpoint validation, source-only imports, strict state loading and public-loader readback; no new training run |
| Installed raw-text inference | Offline target-installed version 0.1.1, unrelated Temp CWD, saved default target Cassidy; mono24 PCM16, 120 frames/72,000 samples for A small model can still speak clearly. |

Actual QA ran on 2026-11-12 from 13:19:39 to 13:22:49 UTC. The single skip was
the existing source-symlink fixture because this OS did not permit creating test
symlinks. Warnings concern existing SWIG type metadata and Torch weight_norm
deprecation, not failed tests. Historical 0.1.0 results below remain separate.
No 0.1.1 CUDA optimization or new-language qualification is asserted.

## Tested Base and Installation

Current **0.1.1** wheel qualification:

| Item | Tested scope |
| --- | --- |
| Wheel | Final metadata revision r002: inflect_mini_v3_adapt-0.1.1-py3-none-any.whl, 47,556 bytes |
| Build | Existing setuptools 70.2.0 vendored wheel backend; no build-dependency download or environment upgrade |
| Installation | Offline `--no-index --no-deps --no-compile --target`; existing qualified dependencies |
| Provenance | Import resolved to the isolated target and reported package0.1.1; Torch CUDA remained uninitialized |
| Working directory | New unrelated Temp directory, outside the source checkout |
| Installed export/inference | Actual CPU export followed by raw-text generation, mono24 PCM16, 120 native frames |
| Contents | 16 entries, Python bytes matched source pins, Apache LICENSE retained; no model/audio/test/dataset assets |
| Limit | Not a fresh complete environment, platform matrix, human listening result or foreign-language quality claim |

Verified 0.1.1 wheel SHA256:

```text
07c917b0dee6eb4961947ff032c0bfd5b709e7adf9b49e4e7614ed1d949272d5
```

The raw-text output was finite and below full scale. This is a functional
contract check, not a naturalness score. The maintainer's source, phase, wheel
and installed-output receipts identify the full-suite run as
`toolkit-qa-011-r001`. After the reviewed README/link metadata changes, the
distinct final wheel's offline installation, provenance, export and raw-text
checks passed again in `toolkit-qa-011-r002`, 13:49:11-13:50:21 UTC. Python
implementation bytes were unchanged; the full CPU suite was not rerun or
double-counted. The earlier r001 wheel remains historical candidate evidence:
`be9e9cf29a4f117def5c37d05694225d9c5747436032f0c7614f7a14982cc654`.

### Historical 0.1.0 Installation

The following **0.1.0/r004** matrix and original wheel receipt are preserved,
not replaced by or relabelled as the 0.1.1 artifact:

| Item | Tested scope |
| --- | --- |
| Package | inflect-mini-v3-adapt 0.1.0; wheel build r004 |
| Interpreter | Python3.12 |
| Installation | Isolated --target package directory; existing qualified dependency environment |
| Working directory | Unrelated Temp directory, outside the source checkout |
| Installed-wheel inference | Raw text generated mono24 output, 193 native frames |
| Wheel contents | 16 files, including Apache LICENSE; no model or audio assets |
| Environment limit | Not a freshly downloaded complete environment or a universal platform/dependency matrix |

Verified r004 wheel SHA256:

```text
13281c2fcb8ed18445d9bf386b732af21f3fc0f77882d1f8e80627cb925942e9
```

The [tested requirements snapshot](../../../finetune_v3/requirements-tested.txt)
records the existing CUDA environment's Transformers 5.2.0, NumPy 1.26.4,
SciPy 1.17.1, SoundFile 0.13.1, phonemizer-fork 3.3.2, and espeakng-loader 0.2.4.
Torch is installed separately. The CUDA smoke and the 41-test model API run
used 2.6.0+cu124, with the model API tests executed on CPU. A separate public
CPU inference benchmark used 2.11.0+cpu; that result does not qualify gradients
or training under 2.11. This is a reproduction snapshot, not a complete freshly
downloaded dependency-environment qualification.

The public-core `weights.pt` SHA256 used for verification is:

```text
8460e7d54a2fbf7fb54b3c7be7569c493c852de9a2aa1a0fd9fd9a099de581ed
```

Record the actual source manifest/weights, prepared dataset, frontend, and
adapted artifact hashes for each run. Inventory validity alone is not proof
that an arbitrary local package is the intended official base.

## CLI Integration Checks Before Promoting Examples

The following integration history is for **0.1.0**, including its separate
CUDA optimization/resume checks. It is not a new 0.1.1 CUDA qualification;
current 0.1.1 CPU/full-model and installed-wheel checks are reported above.

| Verification | Result and boundary |
| --- | --- |
| Final combined suite, including real-model integration and notice continuation | Parent controller run completed 07:51 UTC: 180 passed, 1 skipped, 22 warnings in 104.14 seconds; source and published weights unchanged |
| Real-model API | Earlier parent run: 41 passed on CPU with Torch2.6.0+cu124, covering bridge, public-loader parity, migration, checkpoint and export; not linguistic coverage |
| Separate public CPU inference benchmark | Torch2.11.0+cpu; inference evidence only, not gradient/training-suite qualification |
| Separate lightweight suite | Parent run: 138 passed, 1 skipped in 14.93 seconds, covering data/frontend/notices/training/CLI and other non-model tests; not an additional disjoint total |
| Existing v2 toolkit regression | Separate parent run: 39 passed |
| CUDA optimization/resume | Two updates, resume to four, and continuous four completed; small nondeterminism, no bit-exact equivalence claim |
| Prephonemized export/say | Contract-validated export and explicit-phoneme synthesis passed |
| English eSpeak chain | Six generated examples, prepare, CUDA2, licensed export, raw-text say passed; matching engine(1,52,0), default target Cassidy |
| English source-tree raw-text output | 24 kHz, 120 native frames for A small model can still speak clearly. |
| Installed wheel r004 | Isolated --target install and raw-text generation from unrelated Temp CWD passed; 24 kHz, 193 native frames with existing dependencies |

The English fixtures use the exact clock that generated those same waveforms.
They do not align unrelated recordings or provide new-language training
evidence. Reconstruction validation under supplied alignment and free
predicted-duration inference remain different tests.

The final combined result includes actual real-model integration; the earlier
41-model-test and 138-lightweight results describe component scopes, not extra
passes to add to 180. The notice-continuation regression is included in this
final combined run, not an additional pass. Its exact parent log is
`outputs/run/logs/mini-v3-toolkit-final-regression.log`. The completed r004
wheel installation/raw-text check is a separate artifact verification, not
additional suite passes or a fresh complete-dependency environment test.

The real-model tests can skip when neither a local base fixture nor
`INFLECT_V3_TEST_MODEL` is available. An explicitly supplied invalid model path
still fails; it is not converted to a skip. Preserve the distinction between
the lightweight suite and the separate real-model execution rather than
claiming optional skipped integration ran successfully.

## Current Contract

- Real recordings need external boundary-inclusive positive integer phone
  durations and exactly 600 waveform samples per native frame, mono24.
- Training uses one complete unpadded utterance. Initial text/prosody/speaker
  groups train while bridge/generator weights remain frozen. No F0 extraction
  is required by the duration plus multi-resolution spectral objective.
- Vocabulary additions explicitly use genuinely unmapped native178 IDs;
  existing mappings and model dimensions are preserved. The reviewed map and
  initialization identity must match preparation, checkpoint restore, and export.
- Resume and export bind checkpoint SHA256, controller sidecar, and run-contract
  hash. Export also checks the restored step.
- Exported `target_voice` is the CLI default; other slots are refused as
  unqualified. Raw-text eSpeak inference requires the saved engine version and
  language resources, with no English fallback.
- Full source-package license documents and core notices are required. Export
  preserves the original Apache, StyleTTS2-MIT, and other license texts, hashes
  retained files, and recomputes the final manifest checksum. Historical source
  paths/hashes remain provenance, not adapted-weight claims.

See [workflow](workflow.md) for API/data details and [commands](commands.md) for
the user-facing sequence. The toolkit's [LICENSE](../../../finetune_v3/LICENSE)
and [package metadata](../../../finetune_v3/pyproject.toml) use an Apache license
file and setuptools>=68-compatible license-file metadata.

## Existing V2 Toolkit: Useful Precedent, Not a V3 Loader

The [v2 toolkit contract](../../../finetune/CONTRACT.md) describes a separate
fixed-voice, single-language VITS warm start. Its `model.pth`/`config.json`
checkpoints, posterior/discriminator stages, embedding-extension mechanism,
and `micro-12gb`/`nano-8gb` presets are not interchangeable with Mini v3.
Its corpus and frontend documentation can inform preparation, but use the
v3 [row contract](workflow.md#prepared-data-contract) and commands here.

## Remaining Release Gates

1. Retain the 0.1.1 source/wheel/installed-output receipts with the reviewed
   publication allowlist. Keep 0.1.0/r004 evidence distinct. A changed package
   README, metadata or Python source requires a rebuilt wheel and matching
   installed-target checks; do not transfer a tested hash to changed bytes.
2. Verify the final artifact's inventories, retained licenses, frontend identity,
   target voice, and installation on every environment actually claimed.
3. For a new language, establish genuine external alignment, training-only
   phone/context coverage, and independent development/final-test groups.
4. Select a pilot checkpoint on held-out predicted-duration and raw-text output;
   obtain fluent-speaker review of pronunciation, endings, naturalness, and voice.
5. Publish only the tested runtime/voice/language scope, data rights and known
   limitations. Retained English, other voice slots, events, tuner, enhancer,
   or multilingual behavior require their own evidence.

No language-quality approval follows automatically from installation or smoke
test completion.
