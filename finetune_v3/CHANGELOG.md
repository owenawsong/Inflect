# Changelog

## 0.1.3 - Bounded Windows JSON Replacement

- Use per-call UUID/pid pending paths with exclusive creation, avoiding a
  shared pending filename between concurrent writers.
- Retry only Windows os.replace errors5/32/33:8 attempts,7 delays totalling0.81s.
  Other OS/errors fail immediately; persistent sharing errors still fail.
- Clean only the current writer's owned pending file on success/failure.
  Preserve prior committed JSON on failure; do not delete another writer's temp.
- No model/loss, schema, execution-profile, dependency or inference change.
  Add isolated IO policy/UTF8/cleanup/concurrency tests. This is bookkeeping,
  not a language-quality gain or automatic checkpoint recovery.

Final long-README wheel680c13 passed the installed CPU236/1skip suite, including
42 native,20 publication and17 IO cases. One strict default-four-group CUDA
update on RTX3060 plus CPU export/native24 raw-text WAV passed in29.766s.
The two earlier failed test/fixture attempts remain documented. No final0.1.3
resume, optimizer/WAV-parity or language-quality claim; the completed0.1.2
resume sweep is historical. Exact artifacts and separate private evidence are
identified in [version-specific scope](QUALIFICATION.md).

## 0.1.2 - Training Execution Profile (2026-11-12)

- Express the existing multi-resolution STFT reflect-center values through
  slice/flip/concatenation and center=False, retaining FFT/hop/window/objective.
- Select an equivalent parameterless `(1,0)` left-reflection pad only at the
  recognized training-bridge generator path; unexpected type/padding/state fails
  closed. No global Torch replacement, source-weight, public inference class or
  inference schema change.
- Bind `native24-reflect-cat-v1` in new controller contracts. Reject missing,
  old or altered profiles before weight loading/updates; matching profiles still
  require exact full-contract/sidecar equality. Legacy inference export remains
  subject to its original strict contract and checkpoint checks, not a resume waiver.
- Final metadata wheel70d94a passed installed publication CPU219/1skip,
  including42 native and20 new publication cases. Strict default-four-group
  CUDA SMOKE1/MAIN8 passed with exact full checkpoint/optimizer payload and CPU
  export/raw-text WAV equality. Group trainability is not a claim of nonzero
  gradients for every selected parameter. Existing dependencies and English
  mechanical fixtures only; no fresh0.1.2 environment or language-quality claim.
- Retain private-wheel2d402b speaker-only results, strict0.1.1 CUDA failure and
  separate fresh0.1.1 Windows CPU qualification as historical evidence, not
  additional tests/updates or relabelled final-wheel results.

See [version-specific scope](QUALIFICATION.md) and the guide's
[limits](../docs/v3/language-adaptation/implementation-status.md#012-execution-profile-and-limits).
No tests are relabeled from 0.1.1 and no gradient/checkpoint tolerance is introduced.

## 0.1.1 - Verified CPU Fixes (2026-11-12)

- Load inventoried model modules from freshly hash-verified source bytes instead
  of cached bytecode or native-extension fallbacks. Leave existing caches
  untouched and restore scoped imports afterward. Keep the model code directory
  out of general dependency lookup so unlisted files cannot shadow dependencies.
- Bind external alignment to exact final codepoints after an explicit phone
  map. Do not apply a second NFC/whitespace transformation that changes mapped
  combining marks or token counts. Unmapped normalization is unchanged.
- Add cache, package/helper, dependency-shadowing, mapped timing and small
  checkpoint regressions. No intentional model architecture, checkpoint-format,
  dependency, native clock, gain or language-quality change is introduced.

Full CPU regression: **199 passed, 1 skipped, 23 warnings in 139.15s**, including
42 real-model cases and the combining-map migration/checkpoint/export test.
Earlier 66-targeted and 157-lightweight checks overlap with that scope.

The new 0.1.1 wheel passed offline isolated-target installation, import/version
provenance, CPU export of the existing English mechanical checkpoint, and
raw-text inference from an unrelated Temp CWD: mono24 PCM16,120 native frames.
This used existing qualified dependencies, not a fresh dependency installation,
and establishes no Polish, other-language or human listening quality.

Final metadata revision r002 was rebuilt and its offline isolated-target
import/export/raw-text checks repeated on CPU. Python implementation bytes were
unchanged, so the 199-pass full suite was not rerun for this documentation change.
Tested final wheel: 47,556 bytes, 16 entries, Apache LICENSE, no model/audio assets.
SHA256:

```text
07c917b0dee6eb4961947ff032c0bfd5b709e7adf9b49e4e7614ed1d949272d5
```

The earlier 0.1.1/r001 candidate wheel and its full-suite receipts are retained
unchanged (wheel SHA256
`be9e9cf29a4f117def5c37d05694225d9c5747436032f0c7614f7a14982cc654`).

This is import-integrity hardening, **not a complete sandbox**. Inventoried model
source and installed dependencies remain trusted executable code with the user's
permissions. See the [security notes](../docs/v3/language-adaptation/security-notes.md).

## 0.1.0 - Historical Verification

The r004 wheel passed isolated-target raw-text generation from an unrelated Temp
directory using existing qualified dependencies, not a fresh dependency setup.
Its SHA256 remains:

```text
13281c2fcb8ed18445d9bf386b732af21f3fc0f77882d1f8e80627cb925942e9
```

That artifact and its 180-pass model-inclusive source-suite evidence remain
separate historical records. The 0.1.1 results above come from its own completed
run and newly tested wheel, not from relabelling 0.1.0 evidence.
