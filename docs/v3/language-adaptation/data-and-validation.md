# Data, Coverage, and Leakage-Safe Validation

Prepare clean, well-covered speech with independent evaluation material before
optimization. The v3 [aligned-row contract](workflow.md#prepared-data-contract)
and [command recipes](commands.md) define the supported path. V2-specific
behavior is identified explicitly below so it is not mistaken for a v3 schema.

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
language. Multiple nonempty speaker values are rejected by the v2 preparer;
multi-speaker v3 support must be established from its own implementation.

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

The v2 implementation accepts WAV input and writes 24,000 Hz mono PCM16 copies,
using polyphase resampling when required. Its conversion clips samples to the
configured peak range and records source clipping indicators. Thus a bounded
converted peak does **not** prove the original was unclipped. Low-level format
acceptance is not a speech-quality gate. See
[audio.py](../../../finetune/inflect_finetune/audio.py).

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

The existing v2 manifest minimally accepts UTF-8 JSONL or CSV with `audio` and
`text`. For example, these are illustrative rows, not supplied recordings:

```jsonl
{"audio":"session-01/0001.wav","text":"This is an example sentence.","speaker":"voice-a","group_id":"session-01"}
{"audio":"session-02/0001.wav","text":"Another sentence comes from a separate session.","speaker":"voice-a","group_id":"session-02"}
```

V2 also recognizes `id`, `session`, and `phonemes`; prephonemized mode requires
`phonemes` on every row. Do not assume arbitrary ledger fields survive
preparation. Preserve the ledger separately and reconcile prepared rows against
it. These are **v2** examples and are not training-ready v3 rows: v3 also needs
an audio checksum, exact full phonemes, and externally established integer
durations including both boundary tokens. Never fabricate duration labels to
turn a transcript-only corpus into an aligned one.

## Freeze Split Boundaries Before Optimization

Use three roles: training for weight updates, development/validation for model
selection, and an untouched final test for the final report. The v2 preparer
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

V2 joins identical normalized, case-folded transcripts and explicit split
groups before splitting deterministically. A row's `group_id` takes precedence
over its `session`; the two are not automatically combined. To hold out whole
sessions, put that stronger boundary in `group_id`. Exact duplicate converted
WAV content is rejected. These checks do not detect all overlapping crops,
different encodes, paraphrases, or translations. Review lineage and similarity
separately. See [prepare.py](../../../finetune/inflect_finetune/prepare.py).

Do not break a real source group to force a desired split percentage. If every
row is connected to one source session, collect independent sessions. Repeating
the same sentence many times does not create independent test coverage.

The v3 preparer groups connected rows by explicit `group`, normalized text,
source/prepared PCM identity, and `source.recording_id` when supplied. Its
auditor rechecks those relationships. The controller separately checks exact
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

The v2 audit checks vocabulary membership across prepared splits. It does not
produce a complete training-versus-validation linguistic-phone coverage report.
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
