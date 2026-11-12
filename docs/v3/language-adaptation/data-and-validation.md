# Data, Coverage, and Leakage-Safe Validation

This page describes **Mini v3** `prepare` and `audit`, not the v2 VITS pipeline.
The supported path is licensed recordings plus an explicit frontend and genuine
external timing labels, then preparation, readback audit, and training. The
trainer consumes prepared mono24 audio on its native clock. The preparer has
explicit, recorded resampling/clock opt-ins; neither command automatically aligns
raw recordings. There is no bundled aligner or linguistic-phone-to-model-label
converter. No Polish or other new-language quality result is implied.

## Mini v3 Quick Path

Use the [command recipes](commands.md) for installation and the full workflow.
The following assumes a verified full Mini package, an existing
`./work/adaptation` directory, and genuinely aligned source rows. `fr-fr` is an
example explicit language identity, not a trained language preset.

```sh
python -B -m finetune_v3 vocab --model ./inflect-mini-v3 --output ./work/adaptation/base-vocab.json
python -B -m finetune_v3 coverage --manifest ./data/aligned.jsonl --vocab ./work/adaptation/base-vocab.json --language fr-fr --frontend prephonemized --output ./work/adaptation/coverage.json
python -B -m finetune_v3 prepare --manifest ./data/aligned.jsonl --audio-root ./data/audio --output ./work/adaptation/prepared --vocab ./work/adaptation/base-vocab.json --language fr-fr --frontend prephonemized --split-seed 2026
python -B -m finetune_v3 audit --dataset ./work/adaptation/prepared/dataset.json
```

Stop on unknown coverage before preparing/training. A completed `coverage`
command can report `explicit-vocabulary-migration-required`; it does not assign
IDs, validate timing or inspect audio. If additions are needed, review the
[frontend/vocabulary contract](frontends-and-symbols.md) and supply the same
reviewed vocabulary/map to preparation. Do not delete unknown phones to pass.

Outputs must be new and their parent directories must exist. Raw audio and
prepared output roots must be disjoint. Audio paths inside the manifest are
contained forward-slash relative paths; absolute paths, `..`, backslashes,
drive/stream colons, symlinks and junctions are rejected.

## Mini v3 Source Rows

Input is **UTF-8 JSONL**, one object per line, not CSV. Blank rows, duplicate
JSON keys and non-finite JSON numbers are rejected. Use canonical fields:

| Field | Current preparation contract |
| --- | --- |
| `id` | Unique nonempty string; normalized whitespace/NFC, case-insensitive uniqueness, at most200 characters |
| `audio` | Relative local file below `--audio-root`, readable by SoundFile |
| `text` | Exact nonempty transcript; normalized text is retained alongside `raw_text` |
| `group` | Explicit recording/session/family leakage group; not v2 `group_id` or `session` |
| `source`, `consent` | Nonempty objects containing truthful user-supplied provenance and permission assertions |
| `phones` | Full input phone string for `prephonemized` mode; `phonemes` is an alias, and both must agree if supplied |
| `alignment_phones` | Exact final phone string bound to timing; already-final caller phones can supply it implicitly in prephonemized mode |
| `alignment_source` | Object with matching `language`, nonempty `source`, and `kind`: `human`, `external_forced_alignment` or `dataset_alignment`; `base_predicted: true` is forbidden |
| `phone_durations` or `alignment` | Genuine boundary-inclusive integer frame counts, or genuine timed intervals as described below |

Optional `language` must match the CLI language. Optional `audio_sha256` (or
`sha256`) is checked against the source file if provided; preparation computes
and records the source checksum regardless. The **prepared** audio checksum is
written automatically and required on readback. A source checksum does not
replace recording/session lineage or consent evidence.

Optional `source.recording_id` joins clips from the same original recording
before splitting. Optional `voice` and `speaker_id` are retained metadata,
**not** per-row training voice selection: the current trainer applies its one
`--voice` target to all rows. Arbitrary ledger fields are not guaranteed to
survive preparation; retain the original ledger separately.

### Boundary and Clock Rules

The model consumes **Unicode codepoints**, not a list of linguistic IPA units.
A multi-codepoint phonetic representation needs labels for those exact model
tokens. Do not assume an aligner's one-phone intervals can be split into these
labels automatically; no such converter is supplied.

For `P` final codepoints and duration counts `d`:

```text
IDs = [0, vocabulary[phone0], ..., vocabulary[phoneP-1], 0]
len(d) = P + 2
each d is an integer >= 1, not bool or float
F = sum(d)
prepared samples = 600 * F, at 24000 Hz
```

Both zero-ID boundaries require explicit real timing; they are not optional
padding labels. At24 kHz one frame is25 ms. Final phones contain1..510
codepoints. The preparer bounds total frames to2..1200, but positive counts for
every token also require `F >= P + 2`. Training separately enforces its configured
cap, default480 frames/12 seconds. These are implementation bounds, not
recommended lengths or a foreign-language training recipe.

### Synthetic Schema Example From Tests

These two rows mirror the **synthetic tone fixtures** in
[test_data.py](../../../finetune_v3/tests/test_data.py), with fixture vocabulary
`{"a":1,"b":2," ":3}` and2400-sample/0.1-second files. They are not speech,
phonetic ground truth or reusable real-recording labels. The test's `human`
alignment kind exercises an authored-label parser branch; it does not certify
human approval. Do not transplant these counts or rights assertions onto a
real recording. Two independent components are needed even for this fixture.

```jsonl
{"id":"id-0","audio":"0.wav","text":"Authored sentence number 0.","phones":"a","phone_durations":[1,2,1],"group":"session-0","source":{"description":"Local authored fixture"},"consent":{"declaration":"fixture author"},"alignment_source":{"kind":"human","language":"fr-fr","source":"authored test alignment"}}
{"id":"id-1","audio":"1.wav","text":"Authored sentence number 1.","phones":"a","phone_durations":[1,2,1],"group":"session-1","source":{"description":"Local authored fixture"},"consent":{"declaration":"fixture author"},"alignment_source":{"kind":"human","language":"fr-fr","source":"authored test alignment"}}
```

The fixture encodes `a` as `[0,1,0]`; counts `[1,2,1]` total4 frames and2400
samples. These IDs belong to the fixture vocabulary, **not** a suggested edit
to the published model vocabulary.

### Timed Alignment Instead of Counts

The same unit test replaces `phone_durations` with these intervals, in seconds:

```json
{"alignment":[{"token_id":0,"start":0,"end":0.025},{"token_id":1,"start":0.025,"end":0.075},{"token_id":0,"start":0.075,"end":0.1}]}
```

This is a fragment to replace the duration field on a complete source row, not
a standalone manifest. Every boundary-wrapped token needs its exact `token_id`
and a finite, positive interval. The sequence starts at zero and is contiguous
within half a24k sample. Its final endpoint must cover the full pre-clock-adjusted
native audio duration within the explicitly admitted clock tolerance. If both
counts and timed intervals are supplied, their resulting counts must agree.

The preparer rounds each cumulative endpoint with
`floor(end_seconds * 24000 / 600 + 0.5)` and takes adjacent differences. A short
interval that quantizes to zero is rejected; the tool does not invent an extra
frame, merge phones, infer silence boundaries or run forced alignment. Review
alignment/segmentation/representation externally when it cannot meet the clock.

In eSpeak mode do not supply caller `phones`/`phonemes` as an override. Bind the
external labels with `alignment_phones` to the exact configured eSpeak output.
For an explicit reviewed phone map, labels bind to the **final mapped
codepoints**, including combining marks, without a second NFC transformation.
See the [version-specific coverage](implementation-status.md#staged-011-coverage)
for the staged0.1.1 fix and remaining integration gates.

## Prepared Output and Readback Audit

`prepare` writes a new directory containing `dataset.json`, `dataset.jsonl`,
`train.jsonl`, `validation.jsonl`, `vocab.json`, `audio/<hashed-id>.wav` and
`audit.json`. Audio copies are mono24 **floating-point WAV**, not a PCM16 export.
Source files remain unchanged. Source/prepared checksums, PCM identities, labels,
frontend policy, split assignment and preprocessing details are recorded.

`audit` rechecks the prepared index, hashed row/split/vocabulary files, audio
hashes and decoded PCM, sample/frame clocks, phone IDs/counts, provenance fields
and connected-component split integrity. Both splits must be nonempty. It does
not rerun a human annotation or establish timing accuracy, consent, speaker
identity, fluent pronunciation or generalization. A passing structural audit
is necessary, not proof of usable training speech. Do not hand-edit output rows
and keep an old dataset identity; revise source inputs and create a new version.

## Rights and Corpus Identity

Use recordings whose training use is authorized, with informed speaker consent
and a documented basis for the intended distribution. A base model's Apache
license does not grant rights to somebody else's voice, recordings, texts, or
pronunciation resources. Keep each upstream license and attribution separate.
Record any limits on redistribution of the raw corpus and adapted weights;
do not presume that permission for one implies permission for the other.

Maintain a source ledger outside the model inputs. Include a stable clip ID,
source recording ID and checksum, approved start/end boundaries if segmented,
speaker, session, language/variety, transcript revision, and rights evidence.
Keep the original audio immutable. Converted audio, revised transcripts, and
frontend outputs are versioned derivatives, not replacements for the source.
Do not publish personal paths, consent documents, or private transcripts in a
model package.

For an initial fixed-voice experiment, use one consistent consenting speaker.
Mixing speakers without explicit conditioning can entangle voice identity and
language. Mini v3 preparation does not enforce a one-speaker metadata count;
the current controller still trains one target voice slot for the whole dataset.
Curate that fixed-speaker corpus explicitly rather than assuming row metadata
selects separate conditioning or has been independently verified.

## Record and Segment Deliberately

- Prefer clean, lossless recordings with stable microphone placement, room,
  and recording settings. Repeated sessions should have their own identifiers.
- Avoid overlap, music, corruption, hard clipping, and aggressive processing.
  Review every flagged file rather than trusting a single loudness statistic.
- Segment on linguistic boundaries with complete initial consonants and final
  releases. Do not cut a word in half to fit a convenient duration limit.
- Keep natural within-utterance pauses. Separate non-speech-only material from
  ordinary text-to-speech examples unless the trainer explicitly models it.
- Review stereo before downmixing: averaging channels can cancel speech or mix
  distinct speakers. Select an appropriate source channel when needed, record
  that decision, and preserve the original.
- Keep both short and longer natural sentences within the eventual trainer's
  supported bounds. Choose a duration cap after memory and alignment tests,
  not by copying the enhancer's input limits.

Mini v3 requires nonempty mono input, finite decoded samples and peak strictly
below full scale. It never automatically downmixes, repairs gain or clips unsafe
audio into range. Passing that check does not prove the source was never clipped
or is good speech. A flat/clipped recording can already have a sub-full-scale
peak; preserve originals and inspect quality separately.

By default input must already be24 kHz and exactly on the600-sample clock.
Optional preprocessing is explicit and recorded:

| Option | Actual Mini v3 behavior |
| --- | --- |
| `--allow-resample` | Permit non-24k mono input; use SciPy `resample_poly` into24k and record source rate/count, method and any one-sample resampling-length rounding trim |
| `--clock-tolerance-samples N` | Permit nearest600-sample clock adjustment within integer0..300 post-resampling samples; default0 |
| Row `trailing_silence_samples` | Required for clock trimming; declared24k-sample tail must cover the removed samples and the removed audio must be at or below-55 dBFS |

Resampling does not itself permit off-clock audio. Clock padding adds only the
bounded zero tail; trimming a non-silent tail fails. Converted audio is checked
again for finite, sub-full-scale output. Source/prepared counts, rate, rounding,
trim/pad and tolerance are retained. The input-duration guard is30.025 seconds;
prepared output must still fit1200 frames/30 seconds and the trainer's chosen
cap. Neither resampling nor clock adjustment creates or reassigns phone labels.

An output 48 kHz enhancer is not a data-preparation requirement or a reason to
upsample the training corpus. The current v3 controller requires already
prepared mono 24 kHz audio whose length exactly matches the external alignment;
it does not resample or silently crop a mismatch. See [aligned rows](workflow.md#prepared-data-contract).

## Transcripts Must Match the Recording

Manually verify what was actually spoken, including repetitions, omitted words,
names, inflections, and numbers. ASR is an annotation aid, not ground truth.
Keep the original orthographic text and the exact normalized spoken form.
Do not silently make the transcript more grammatical than the recording.

Define conventions for punctuation, quotes, abbreviations, numerals, units,
foreign words, and non-speech. Do not insert stage directions into `text` unless
the chosen frontend and training objective explicitly support them. A sentence
with the wrong phonemes teaches the wrong text/audio correspondence even when
the waveform is clean.

The v3 text frontend uses NFC/whitespace normalization with no implicit English
number-expansion policy. In prephonemized mode supply the reviewed phones
yourself; in eSpeak mode use the exact explicit language/frontend and reject
switch/fallback flags or unknown phones. Preserve original text, frontend output
and the ledger so a later change can be traced back to the recording. Neither
`coverage` nor a readable transcript supplies the missing external timing labels.

## Freeze Split Boundaries Before Optimization

Use three roles: training for weight updates, development/validation for model
selection, and an untouched final test for the final report. The Mini v3 preparer
creates train and validation only; maintain a final-test reservation separately,
outside the manifest supplied to preparation. A validation clip repeatedly
consulted during tuning is not an untouched test.

1. Group all related material first: crops from one recording, overlapping
   windows, alternative encodes, retakes, duplicate or near-duplicate prompts,
   and any derived versions. Keep every connected family in one partition.
2. Choose the generalization boundary you care about. Holding out recording
   sessions tests more than a random clip split. Holding out books or prompt
   families helps avoid template and narrative-context leakage.
3. Reserve sufficiently varied development and test material. Report counts,
   usable duration, independent groups, and coverage, not just a percentage.
4. Freeze the split manifest and hashes. Any later data change gets a new
   dataset version and a disclosed re-evaluation plan.

Mini v3 builds transitive connected components using normalized `group`,
normalized/case-folded content transcripts, source PCM identity plus source rate,
prepared PCM identity, and normalized `source.recording_id` when provided.
For example, a session link plus a repeated transcript can join otherwise
different files into one component. Linked rows must never cross splits.

Without explicit row `split` values, the seed deterministically ranks component
IDs and assigns validation components. `validation_fraction` is a fraction of
**components**, not guaranteed clips or hours. With explicit splits, every row
must supply `train` or `validation`, each component must stay in one split, and
both splits must be nonempty. A row marked `test` is not a supported third split;
keep final-test recordings outside this preparation manifest.

Do not break a real source group to force a desired split percentage. If every
row is connected to one source session, collect independent sessions. Repeating
the same sentence many times does not create independent test coverage.

The readback auditor rechecks those relationships. The controller separately checks exact
audio-hash overlap and case-folded, whitespace-normalized transcript overlap.
These checks do not establish the absence of every overlapping crop or semantic
near duplicate. Keep lineage review and a separate final-test reservation.

## Coverage Is More Than an Inventory Check

Create a training-only coverage table after phonemization. Count symbols and,
where the frontend allows, linguistic phones, adjacent phone pairs, word
positions, stressed/unstressed contexts, sentence types, and length ranges.
Track the number of **independent clips and sessions** containing each item,
not only total occurrences. Repetition in one long recording can hide gaps.

Keep these three questions separate:

| Check | What it establishes | What it does not establish |
| --- | --- | --- |
| Every character is in the model vocabulary | Input can be represented | A linguistic phone is correctly encoded or learned |
| A target phone occurs in training | At least some exposure exists | Enough clean, varied examples or correct timing |
| A contrast is intelligible on unseen text | Evidence for that tested contrast | Universal language quality |

Mini v3 `coverage` reports unknown final codepoints and selected examples;
`audit` checks prepared vocabulary/IDs. Neither produces a complete
training-versus-validation linguistic-phone coverage report.
An added symbol appearing only in validation can pass inventory checks but have
no training examples. Build and review that comparison explicitly.

Resolve missing training coverage while designing the corpus. After freezing
the final test, do not move failing test sentences into training and keep calling
the same set unseen. New coverage material must come from independent sources;
if the test influenced development, retire or relabel that test.

## Evaluate Without Hiding Failures

Predeclare a held-out panel covering ordinary sentences, rare contrasts,
numbers and inflections, names, punctuation, long inputs, and unsupported input.
Use the same texts, frontend contract, output rate, and declared seed policy for
candidate comparisons. Evaluate raw native output first; assess enhancement or
other optional components separately.

Record each attempted item, including failures and nonfinite, empty, silent,
clipped, truncated, or repeated output. Failed synthesis is not a zero-WER
sample; pending evaluation is not a score of zero. Aggregate ASR edit counts
and reference word counts (micro-WER), preserve the normalization policy, and
provide per-item differences. Report CER when helpful for inflected languages,
without treating it as a replacement for word intelligibility.

Have fluent speakers review pronunciation, naturalness, accent, stress, voice
consistency, and omissions using blinded, randomized comparisons when feasible.
Keep diagnostic ASR/MOS results separate from human listening results. Inspect
regressions as well as improvements, and disclose selection or excluded items.
The release report should distinguish raw-text inference, explicit-phoneme
diagnostics, and any acoustic reconstruction tests; they answer different
questions.

## Historical V2 Appendix

V2's CSV/JSONL rows, `speaker` count checks, `group_id`/`session` precedence and
automatic24k mono PCM16 conversion are **not** the Mini v3 contract above. Its
conversion may resample and clip to a configured peak, so bounded converted
amplitude was never proof of unclipped source audio. Do not use a bare v2
transcript row or a v2 checkpoint as v3 aligned training input.

For that separate toolkit, use its [README](../../../finetune/README.md),
[contract](../../../finetune/CONTRACT.md),
[preparer](../../../finetune/inflect_finetune/prepare.py) and
[audio implementation](../../../finetune/inflect_finetune/audio.py). The rights,
immutable-source ledger, independent split design and honest quality-reporting
principles still apply to either toolkit; their schemas and conversion behavior
must not be conflated.
