# Evaluation Plan

Inflect uses listening tests and objective checks. A model has to pass both.

## Blind A/B

Blind A/B is the main human evaluation loop.

Required comparisons:

- base model vs tuned checkpoint
- short prompt vs long prompt
- stable voice vs difficult voice
- plain text vs punctuation-heavy text
- raw output vs enhanced output

Listening priorities:

- speaker similarity
- naturalness
- pacing
- intelligibility
- emotional fit
- stability under long prompts
- absence of clicks, pops, and volume jumps

## Objective Metrics

Objective metrics do not replace listening, but they catch obvious failures quickly.

Planned metrics:

- duration ratio: generated seconds vs expected speaking time
- leading silence
- internal silence
- RMS variance and sudden loudness jumps
- ASR text coverage
- repeated phrase detection
- clipped audio percentage
- real-time factor
- peak memory use

## Stress Prompt Categories

Benchmark prompts should include:

- short neutral sentences
- conversational prompts
- punctuation-heavy prompts
- natural pause prompts
- soft emotion
- excited emotion
- long narrative
- clone-stress prompt
- dialogue-like text
- questions stacked together
- product demo copy

## Pass / Fail Rules

### Good

- voice stays intact
- words are not skipped
- pacing is plausible
- long prompts do not collapse
- emotion helps rather than distracts
- output does not require cherry-picking

### Mid

- voice is mostly intact
- pacing varies but remains usable
- occasional artifacts are audible
- long prompts need runtime chunking

### Bad

- skipped or invented words
- random quiet/loud swings
- slo-mo speech
- repeated phrases
- wrong speaker identity
- broken long prompts
- enhancer hides defects instead of fixing them

## Release Requirement

The public sample gallery should include:

- best samples
- average samples
- known failure cases
- raw and enhanced comparisons

The goal is credibility, not only hype.
