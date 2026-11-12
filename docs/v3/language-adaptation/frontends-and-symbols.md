# Frontends, Custom Phonemes, and Multilingual Limits

Choose a reproducible frontend and a reviewed symbol mapping before aligning
recordings. The same language/phone contract must reach training and inference.
See [commands](commands.md) for coverage and [tested scope](implementation-status.md)
for the distinction between working tooling and evaluated language quality.

## Treat the Frontend as Part of the Checkpoint

The complete contract is raw text, normalization, pronunciation/G2P, punctuation
and boundary conventions, exact symbol-to-ID mapping, and the acoustic model.
Record every stage and use the same contract during training and deployment.
Changing frontend versions after training can change model inputs even when
the visible text is identical.

Store frontend code and dependency versions, language/variety, stress policy,
word and phone separators, dictionaries, external model hashes, Unicode policy,
and any context-dependent number or abbreviation rules. Keep golden examples
with expected normalized text, full phoneme string, and token IDs. Check exact
repeatability in a fresh process, not just a cached call.

## Three Input Modes

| Mode | Appropriate use | Necessary qualification |
| --- | --- | --- |
| Configured eSpeak NG G2P | A supported language with reviewed pronunciation and normalization | Verify the installed language, exact output, stress, and boundaries |
| Prephonemized input | A corpus with reviewed phonemes, or a diagnostic that isolates G2P errors | Supply consistent full strings for all rows and at inference |
| Custom frontend | Language-specific normalization, lexicons, or a different G2P implementation | Trusted reproducible code, vocabulary checks, resource licenses, and deployable dependencies |

These three modes exist in the **v2** toolkit. Their existence does not imply a
matching v3 custom-hook mode. The new v3 frontend implements explicit eSpeak
and prephonemized modes; custom G2P can supply reviewed phones externally, but
no v3 trusted-hook CLI is established here. A phoneme-input export is not a
raw-text system unless the matching frontend is supplied and evaluated.

Phonemizer supports multiple backends with different stress, punctuation, and
boundary behavior; they are not interchangeable. Its API documents language
switch flags and word-count mismatches. Removing switch flags removes the
markers, not necessarily the switched pronunciation. Keep warnings visible
during corpus review and test foreign words explicitly.
[Phonemizer overview](https://bootphon.github.io/phonemizer/),
[API reference](https://bootphon.github.io/phonemizer/api_reference.html).

## Existing V2 Custom Hook Contract

The inspected v2 loader accepts a trusted factory in `module:callable` or
`file.py:function` form. It accepts no arguments or a `language` keyword and
returns an object with:

```python
normalize(text: str) -> str
phonemize(normalized_text: str) -> str
symbols() -> list[str]
metadata() -> dict
```

`symbols()` declares unique, nonempty one-character symbols; metadata includes
`name`, `version`, `language`, and `configuration`. Preparation compares repeated
normalization and phonemization calls, fingerprints source and declared metadata,
and rejects undeclared output characters. Loading the hook executes Python:
review it as code, not as a harmless configuration file. Pin its external
dictionaries and models too; hashing one Python file does not hash everything
that file imports. See [frontend.py](../../../finetune/inflect_finetune/frontend.py)
and [custom G2P documentation](../../../finetune/docs/CUSTOM_G2P.md).

## Unicode and Phone Representation

The v2 default normalizer uses NFKC for raw text and NFC for phoneme output.
Custom-hook output also passes through NFC and whitespace normalization.
Canonical and compatibility normalization serve different purposes; NFKC can
remove formatting distinctions that may affect how a number or symbol should
be read. Retain the original text and test any such transformations.
[Unicode Standard Annex #15](https://unicode.org/reports/tr15/).

The new v3 frontend instead uses NFC plus whitespace collapse for text and
phones, with no English number expansion. Its declared language is mandatory.
The eSpeak path keeps switch flags visible and rejects them rather than
silently removing evidence of a switched pronunciation. Do not assume the v2
normalization/switch policy describes v3.

The inspected models consume character-level symbol streams. A linguistic
phone written with two letters, a combining mark, or an affricate tie is not
necessarily one model token. Decide whether an affricate is represented by a
single code point or a sequence, document that choice, and use it consistently.
Do not replace unknowns by vaguely similar English sounds just to obtain a
passing vocabulary check. Do not silently drop stress marks, diacritics,
combining characters, or punctuation without reviewing the linguistic effect.

V2 preserves the published base inventory prefix and appends new symbols. Its
checkpoint migration matches symbol identity and occurrence, including a
legacy duplicate symbol, rather than arbitrarily reordering IDs. New rows are
deterministically initialized. Initialization provides no acoustic knowledge;
new sounds and contexts still need paired training examples. See
[symbols.py](../../../finetune/inflect_finetune/symbols.py) and
[checkpoint.py](../../../finetune/inflect_finetune/checkpoint.py).

## V3 Mini Is Not an Expandable V2 Export

The inspected v3 Mini deployment is eval-only, checks the exact text-model
configuration, and validates a character vocabulary with IDs in the native
178-ID range. The full phoneme-input helper rejects unknown characters and
overlong strings instead of truncating or falling back. The English frontend
is not a general multilingual selector.

The emerging bridge now exposes experimental `migrate_vocab`: an explicit,
CPU-only, pre-training map from new single-character phones to distinct IDs
that are actually unmapped in the current vocabulary, within 1..177. It keeps
the 178-row factories and old mappings intact, initializes the selected rows in
both embeddings deterministically (or copies explicitly named old phone rows),
and records a migration receipt. It does not infer phonetic equivalence or
provide an unlimited inventory. Existing IDs must never be reassigned.

Apply a reviewed migration before optimizer construction, device movement, or
the first training forward; reconstruct exactly the same migration for
checkpoint restore. Pin the map, seed, initializer policy, retained-row hashes,
and resulting contract. The current `train()` controller now accepts a
`vocab_migration` JSON path and applies its explicit map before optimization;
export reconstructs the same migration from the run contract. An existing CPU
fixture receipt reports retained-row, initialization, checkpoint, and export
checks, not real-language frontend coverage or language quality. Parent
final lightweight data/frontend/notices/training/CLI suite reported 138 passed
and one skipped; that
software result does not certify a language's phonetic transcription or audio.

Adding Polish or another language therefore requires a coordinated frontend,
vocabulary receipt, controller, and export path, not a manual edit to a public
deployment vocabulary. If unused rows are insufficient, a separately specified
architecture/export migration is required. Do not bypass strict checks or claim
drop-in compatibility before actual loader and inference tests.

## Language-Aware Normalization

Create golden examples for ordinary text plus dates, ordinals, decimals,
currencies, units, abbreviations, acronyms, names, inflected forms, loanwords,
quotes, and mixed scripts. Specify the intended spoken reading in context.
The same written numeral may require different spoken forms in two sentences.
Do not apply English number or abbreviation rules globally.

If a corrected full phoneme string improves a diagnostic output, it suggests a
frontend issue worth investigating. It does not establish raw-text performance,
nor does it prove the acoustic model handles every new contrast. Report both
raw-text and phoneme-input paths honestly.

## Multilingual and Code-Switching Work

One language code passed to a G2P backend is not multilingual conditioning.
Language tags, if introduced, need declared IDs, training examples, inference
behavior, and tests. Speaker, language, and recording conditions can become
confounded if each language appears only with a different speaker or channel.

For a future multilingual experiment, specify per-language corpora and sampling,
shared versus distinct symbols, language identification, and whether
code-switching is a target or unsupported. Evaluate every language separately,
shared-phone contrasts, names/borrowings, and mixed-language sentences if
claimed. Re-test retained English behavior if retention is part of the goal.
No multilingual sampling ratio or retention method is established by the
current mechanical tests.

Package the actual inference frontend, resource licenses, and immutable
checksums. Missing target-language resources should produce a clear error,
never a silent English fallback. Test deployment without the training checkout
or source recordings before describing it as self-contained.

CLI export now writes explicit language/frontend identity and vocabulary into
`frontend.json`, together with one `target_voice`; CLI inference refuses other
slots as unqualified. Full raw-text packaging must also retain frontend
mode/version/configuration, any unused-ID mapping, and its resources.
An English convenience frontend must not be inherited silently when integrating
an adapted model core. If a target-language frontend cannot be loaded or its
identity differs, refuse raw-text synthesis. A deliberately phoneme-input-only
artifact must say so and must not advertise raw-text language support.

The CLI now checks the actual eSpeak engine version against the version saved
with the training frontend before raw-text synthesis proceeds. The optional
`smoke-dataset --frontend espeak` uses fixed English (`en-us`) prompts to test
that restore path. Historical0.1.0 verification passed this separate English
mechanical chain through prepare, CUDA two updates, licensed export, and raw-text `say`,
with engine `(1, 52, 0)` matching. It is distinct from the prephonemized smoke
and does not certify foreign-language coverage or speech quality. Independent
wheel installation also passed within its [documented existing-dependency
scope](implementation-status.md#tested-base-and-installation). The separate
[fresh0.1.1 Windows CPU result](fresh-windows-cpu-011.md) does not qualify CUDA training.

Use the new label-free `coverage` command before alignment to inspect unknown
code points and original/final frontend output. Its report can require migration
even when the process exits successfully. It does not create timing labels,
approve the phonetic transcription, or establish language quality.
