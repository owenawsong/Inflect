# Changelog

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
