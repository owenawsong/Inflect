# Polish Adaptation: A Review Plan, Not a Quality Claim

Polish quality is not qualified by the tooling tests. Use this review plan to
design a genuine Polish pilot. Native orthography below is intentional:
stripping diacritics would conceal frontend failures this checklist exposes.
The examples are newly authored review prompts, not supplied audio or an
approved phoneme transcription.

## Establish Feasibility First

eSpeak NG lists Polish as `pl`, and its upstream language definition and rules
exist. This establishes a frontend candidate, not a qualified Inflect language.
Verify the actual installed backend and data version rather than assuming the
upstream language list matches a bundled English-only resource set.
[Language list](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md),
[Polish definition](https://github.com/espeak-ng/espeak-ng/blob/master/espeak-ng-data/lang/zlw/pl),
[Polish rules](https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/pl_rules).

The Mini v3 bridge rejects unknown characters unless an explicit pre-training
unused-ID migration has been applied. The experimental helper preserves the
178-row architecture; an existing CPU fixture receipt is not Polish
qualification. Before recording a large corpus, phonemize a representative Polish
panel and compare **every code point** against the actual vocabulary. If new
symbols are needed, agree on a reviewed mapping to genuinely unused IDs and
bind it to the training and export contracts. The current controller accepts an
explicit migration JSON file before optimization. Do not discard needed
symbols, reuse an existing phone's ID, or edit the public package's pinned
vocabulary to pass.

The [coverage command](commands.md) can perform this text/frontend audit before
recording or alignment: it needs no duration labels. Treat an unknown-phone
report as a real prerequisite to resolve, not as permission to substitute an
English approximation. The separate six-example English smoke fixture is not
Polish coverage, alignment, or perceptual evidence.

A fully representable string is only the first gate. English-trained weights
are not thereby known to distinguish Polish sounds, stress, clusters, or
inflectional endings. Evaluate acoustics separately on unseen Polish text.

## Choose the Target Variety and Speaker

State the intended regional variety, speaking register, and speaker. Have a
fluent reviewer agree on the intended reading and acceptable variants before
annotating the corpus. Do not describe one adapted speaker as all Polish
speakers, or silently mix different conventions into the training labels.

Polish has substantial consonant contrasts and complex clusters; Jassem's
language illustration is a primary phonetic reference. Those properties make
cluster preservation and consonant intelligibility important review targets,
not reasons to infer the size of a successful corpus.
[Jassem, Polish, JIPA 33(1), 2003](https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/polish/8CDD2C699493A69A40A5E7D923629574).

## A Concrete Frontend and Coverage Checklist

The following are **items to verify with a fluent speaker and the chosen G2P**,
not a complete phoneme inventory or a normative pronunciation dictionary.

| Review area | Suggested material | What to inspect |
| --- | --- | --- |
| Diacritics and Unicode | `ą ć ę ł ń ó ś ź ż`, uppercase forms, composed and decomposed input | No accent stripping, replacement characters, or accidental ID reuse |
| Sibilant and affricate distinctions | `s / sz / ś`, `z / ż / ź`, `c / cz / ć`, `dz / dż / dź` in varied words | Consistent distinct representations where required; intelligibility rather than English approximations |
| Contextual spelling | `si`, `ci`, `zi`, `ni` before different vowels and consonants | Context-aware output, not a universal letter substitution |
| Vowels and nasal spellings | `i`, `y`, and `ą`, `ę` in different positions and neighboring contexts | The actual spoken realization agrees with the frontend labels |
| Voicing and boundaries | Word-final consonants, clusters, prepositions plus following words | Contextual behavior and word-boundary handling |
| Dense clusters | Ordinary words and phrases, not only a tongue twister | No inserted vowels, skipped consonants, or swallowed endings |
| Stress and phrasing | Ordinary words, borrowed words, inflected forms, questions, lists, punctuation | Intended word stress and sentence prominence, not imposed English patterns |
| Morphology | Noun/adjective agreement, verb forms, numeral contexts | Correct spoken endings and complete final syllables |
| Names and borrowing | Polish names and place names plus intended foreign terms | Agreed reading, no unnoticed English language switch |

Cover these in independent recordings and varied linguistic contexts. One
pangram does not establish phonetic or prosodic coverage. Preserve the exact
phoneme strings from the selected frontend; the letters in this table are
orthographic reminders, not token IDs or a replacement G2P.

## Numbers Need Context

Use a frontend test sheet with original text, intended spoken text, actual
normalized text, phonemes, and reviewer disposition. Include dates, times,
decimal commas, currencies, ordinals, units, abbreviations, and acronyms.
Inflection and intended reading can depend on the surrounding phrase, so
generic digit expansion is insufficient as a review procedure.

Illustrative prompts, to be reviewed and kept separate from any final test:

```text
Pociąg odjeżdża o ósmej trzydzieści.
Proszę zapisać datę: 12 listopada 2026 roku.
Temperatura wynosi 3,5 stopnia.
Widzę dwa koty. Rozmawiam z dwiema osobami.
Łukasz jedzie do Łodzi, a Zofia wraca z Gdańska.
Czy możesz przeczytać to jeszcze raz?
```

Have the recording and normalized transcript agree on the chosen reading. If
the speaker says a fully expanded form, do not leave a contradictory expansion
in the prepared text. Keep source orthography for traceability and test the
deployed raw-text path on numerical forms as a separate qualification.

## Alignment and Training Are Separate Gates

The inspected v3 bridge requires externally supplied, hard monotonic alignment
that covers all input tokens, including boundary tokens, at its native frame
clock. It does not infer trustworthy Polish duration labels from the transcript
or validate an aligner's language competence. Use an alignment method qualified
for the actual Polish recordings and phone representation; review timing at
clusters, short segments, and utterance boundaries. Low waveform reconstruction
error under supplied alignment is not proof that predicted durations work.

New-language adaptation is also not the same task as moving an English voice
embedding. Start with the actually implemented trainable groups, determine what
supervision is available, and evaluate the resulting full text-to-audio path.
There is no qualified Polish freeze schedule, learning rate, or update count in
this guide. The current controller starts with text/prosody/speaker groups
trainable and the bridge/generator frozen, using supervised duration and
multi-resolution waveform spectral loss. It requires no extracted F0 labels.
That is an implementation fact, not Polish quality qualification. See the
[current workflow](workflow.md) and [pilot and training](pilot-and-training.md).

## What a Polish Evaluation Should Say

Maintain held-out ordinary prose, difficult contrasts, numbers and inflections,
names, longer phrases, and natural punctuation. Keep source-session and prompt
family boundaries intact. Include failure and regression cases, not only
successful demonstrations.

Use a Polish-capable ASR evaluator only as a diagnostic; pin its version and
normalization, report word/character errors with reference counts, and inspect
disagreements manually. ASR may favor fluent-looking text while missing an
accent or a consonant contrast. Fluent-speaker listening must assess the
pronunciations, stress, naturalness, complete endings, and speaker identity.
Keep automatic scores and human results separate.

Until that evidence exists, an accurate description is **"experimental Polish
adaptation pipeline; output quality not yet qualified"**, not "Polish supported"
or "multilingual." If code-switching, retained English, enhancement, or the
original six voices were not tested after adaptation, do not imply they are
preserved.
