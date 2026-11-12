# Import Integrity and Trust Boundary

## Version 0.1.1

The toolkit's scoped model importer reads inventoried Python sources, verifies
their exact bytes and compiles those bytes directly. It does not read or write
the model modules' bytecode caches, delete user caches, or fall back to unlisted
owned bytecode/native extensions. The model code directory is excluded from
general dependency search during that scope. Existing modules, paths and
finders are restored afterward.

This addresses unverified bytecode executing instead of inventoried source,
including valid timestamp/unchecked-hash caches and unlisted dependency shadows.
It is **not a complete sandbox**, a global cache policy, or a guarantee against
malicious installed dependencies/import hooks. Inventoried model source still
executes with your account's permissions. Hashes checked against a supplied
manifest show consistency with that manifest, not independent publisher
authenticity. Obtain code/weights from a trusted source and verify the intended
published pins; do not load an arbitrary untrusted model package.

## Alignment Integrity

An explicitly reviewed phone map produces final codepoints after the initial
NFC/whitespace normalization. Alignment labels must match those exact final
codepoints: NFC-equivalent strings are not interchangeable when they tokenize
differently. The fix does not guess timing, assign vocabulary IDs, or generate
alignment labels. Original unmapped normalization and external-label provenance
requirements remain in place.

## Qualification

The full CPU suite passed 199 tests, with one platform-dependent skip, including
42 real-model cases. The final 0.1.1 wheel also passed isolated-target import,
export and raw-text checks using existing qualified dependencies. These checks
do not establish new-language quality or a fresh dependency environment. See
[tested coverage](implementation-status.md#staged-011-coverage).


Separate **0.1.1 fresh Windows CPU qualification** subsequently passed dependency installation/provenance and the installed prepare/audit/train-one-update/export/raw-text chain. It used Python3.12.11, Torch2.6.0+cpu and eSpeak1.52.0; it is distinct from the earlier offline-target checks and does not rerun their full pytest suite. See [exact fresh-install environment, commands and limits](fresh-windows-cpu-011.md).

The strict0.1.1 CUDA backward failure remains a failure; CPU qualification is not a CUDA workaround or a language-quality approval. Final-wheel0.1.2 strict model CUDA checks passed the default four-group configuration within the documented Cassidy/English-mechanical scope; this is not a new-language quality claim.
