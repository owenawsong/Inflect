# Keep Phone Representation and Vocabulary Migration Consistent

Start with the extracted `base-vocab.json` and its reported `unused_ids`.
Run label-free coverage before audio alignment. Unknown codepoints require a
reviewed decision; successful command exit is not permission to drop or
transliterate them. No Polish equivalences or automatic ID choices are supplied
here. A small coverage panel cannot establish a complete language inventory.

Three different files have different jobs:

| File | Meaning |
| --- | --- |
| `pl-phone-map.json` | Optional, reviewed one-codepoint representation replacements; no deletion, expansion or sound-equivalence guessing |
| `reviewed-vocab.json` | Complete base character-to-ID map plus explicitly admitted new characters; preserve every existing mapping |
| `pl-migration.json` | Only the new-character-to-unused-ID assignments and deterministic initialization settings used by training/export |

The following are **symbolic templates, not runnable configurations**.
Replace every angle-bracket placeholder with reviewed values. An ID placeholder
must become an integer from the actual extracted unused-ID list. A phone
placeholder must become exactly one Unicode codepoint, not a phone name.

```json
{"language":"pl","name":"<REVIEWED_RULESET_NAME>","replacements":{"<ONE_RAW_CODEPOINT>":"<ONE_FINAL_CODEPOINT>"}}
```

If no representation change is needed, `replacements` may be `{}`. A reviewed
no-op map does not solve an unknown vocabulary character. Use the same map for
coverage and preparation; eSpeak export records it for raw-text inference.

```json
{"phone_to_id":{"<ONE_REVIEWED_NEW_CODEPOINT>":"<UNUSED_ID_INTEGER>"},"seed":2026}
```

`phone_to_id` is additions only. Build the complete prepared vocabulary as
`reviewed_vocab = base_vocab + phone_to_id`, retaining all base keys and IDs.
Its JSON file itself is a plain character-to-integer object, not the migration
object above. Every addition needs a distinct previously unused ID in1..177;
0 remains boundary-only. No public package file is edited.

The seed above is an explicit initializer example, not a language-training
recommendation. Omit optional `copy_from` unless an explicitly reviewed copy
is justified. Its schema is `{new_phone: existing_base_phone}`: values are
existing phone characters, not numeric IDs. It cannot chain new additions.
Without a copy, deterministic random embedding initialization provides no
phonetic knowledge or guarantee of useful acoustics.

Check the full contract before annotation/training:

1. Every raw frontend codepoint is retained or changed by an explicit reviewed
   one-to-one rule. New contrast distinctions must not be erased to pass coverage.
2. The final mapped string is covered by `reviewed-vocab.json`.
3. `--base-vocab base-vocab.json` verifies old mappings are unchanged.
4. The migration's additions produce exactly that complete prepared vocabulary.
   Training otherwise rejects `Prepared vocabulary differs from model`.
5. External labels bind to the exact final mapped `alignment_phones`, including
   both zero-ID boundary tokens; do not apply a second normalization afterward.
6. Use the same migration file for initial training and resume. Export reconstructs
   its saved mapping from the run contract; it does not choose IDs again.

If no additions are needed, keep the base vocabulary unchanged and omit the
entire `--vocab-migration` option. If no phone map is needed, omit that option
consistently, rather than introducing an unexplained transform.

See the [real source-row and timing schema](data-and-validation.md#mini-v3-source-rows).
External alignment is mandatory. Coverage does not create durations, align
recordings, train a model, or approve pronunciation.

## A Small Measured Polish Codepoint Audit

On2026-11-12, an existing eSpeak NG1.52.0 backend processed
[12 authored Polish prompts](evidence/polish-coverage-20261112/prompts.jsonl)
in a label-free CPU coverage check against the114-character native base
vocabulary. The installed DLL and Polish resource files were pinned. The
[sanitized summary and pins](evidence/polish-coverage-20261112/summary.json)
record the exact scope and source-report identity.

The report found **U+0291 `ʑ` eight times** outside that base map and returned
`explicit-vocabulary-migration-required`. Eight is a codepoint occurrence
count, not eight pronunciation errors or eight failed sentences. No phone was
replaced, dropped or assigned an ID. This result is a reason to review the
representation and any required vocabulary extension, not a recommendation
to map it to an English sound.

This is **coverage evidence only**, not Polish training or support qualification.
It used no recordings, speech generation, neural model weights or timing labels,
and had no fluent-speaker pronunciation review. Representable output can still
be linguistically or acoustically wrong. The small prompt panel does not prove
a complete Polish inventory, nor does a functioning G2P backend qualify an
adapted speaker, raw-text acoustics or unseen text.
