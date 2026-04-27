# Contributing

Inflect is a research-preview project. Contributions are welcome, but the bar is evidence, not vibes.

## Good Contributions

Useful contributions usually fall into one of these categories:

- reproducible benchmark improvements
- clearer docs
- safer data preparation
- better evaluation prompts
- runtime stability fixes
- small, well-scoped model experiments
- bug reports with audio examples and exact commands

## Before Opening a PR

Please make sure:

- the change is small enough to review
- generated artifacts are not committed
- private reference voices are not committed
- checkpoints are not committed
- local absolute paths are not added to public docs
- the README does not claim unreleased model quality

## Experiment Reports

For model or inference changes, include:

- base variant
- changed variant
- exact command
- prompt set
- voices tested
- checkpoint used
- listening notes
- objective metrics if available

Minimum listening notes:

- speaker similarity
- pacing
- skipped words
- glitches
- long-prompt behavior

## Commit Scope

Keep PRs focused.

Good:

- "Add ASR pseudo-label filter"
- "Improve README and media kit"
- "Add duration-ratio check to benchmark"

Bad:

- one PR containing docs, checkpoints, generated audio, unrelated training changes, and local state files

## Local Artifacts

Do not commit:

- `outputs/`
- `.blind_ab_state*/`
- `reference_voices/`
- `ZipVoice-official/`
- `third_party/`
- checkpoints
- full generated datasets
- private audio

## Project Tone

Inflect should be ambitious without exaggerating. If a model is experimental, say it. If a sample uses an enhancer, label it. If a method failed, document the failure.
