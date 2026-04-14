# LuxTTS Port Checklist

Last updated: 2026-04-12

---

## Goal

Port the parts of LuxTTS that are actually good for Inflect, and reject the parts that are only opinionated or harmful.

This is not "copy Lux wholesale."

This is:
- identify real Lux improvements
- keep the useful ones
- reject the bad defaults
- build a cleaner Inflect runtime on top of `zipvoice_distill`

---

## Confirmed Lux Changes

These are real code-level differences from stock ZipVoice.

### 1. Distill-first path

Lux is built around:
- `ZipVoiceDistill`
- usually `4` inference steps

Assessment:
- good
- this matches the current Inflect speed/quality direction

Decision:
- keep

### 2. Custom solver update

Official ZipVoice:
- plain Euler update

Lux:
- predicts `x_0` and `x_1`
- uses anchored probability-flow style update
- snaps to `x_1_pred` on the final step

Assessment:
- real
- modest effect by ear so far

Decision:
- maybe keep
- only if it survives broader listening

### 3. Custom LinaCodec / Vocos 48k decoder

Lux decoder is not just stock Vocos.

It uses:
- normal 24k branch
- `upsampler`
- `head_48k`
- crossover merge between 24k and 48k branches

Assessment:
- real
- strongest Lux-side component difference
- best candidate for audible improvement

Decision:
- keep investigating
- this is the main Track A component

### 4. Direct prompt encode / direct generate wrapper

Lux uses:
- direct prompt encoding
- direct generation path
- less chunking / stitching than stock ZipVoice inference

Assessment:
- useful for deployment simplicity
- not obviously superior in quality by itself

Decision:
- maybe keep pieces
- do not blindly replace stock inference behavior everywhere

---

## Lux Defaults To Reject

These are not good Inflect defaults.

### 1. `speed = speed * 1.3`

This is in Lux generation code.

Assessment:
- bad default
- makes outputs unnaturally fast
- already caused obviously wrong local behavior

Decision:
- reject

### 2. Aggressive prompt duration cap as a default

Short prompt windows can help inference speed, but they also affect duration estimation and can produce over-fast speech or prompt leakage problems.

Assessment:
- risky
- useful only as an optional test, not a mainline default

Decision:
- reject as a default

### 3. Blind fallback to generic Vocos

Lux checkpoint expects LinaCodec-compatible decoder pieces.

Assessment:
- acceptable only as a debugging fallback
- not faithful Lux behavior

Decision:
- reject as a serious reproduction path

---

## Things Lux Does Not Really Change

Do not over-credit Lux for these.

- `Zipformer2`
- `SwooshR`
- existence of `t_shift`

These are already in ZipVoice or not unique enough to matter.

---

## Current Inflect Direction

### Keep

- `zipvoice_distill`
- `4-step`
- direct transcript use when available
- Track A decoder experiments

### Maybe Keep

- some wrapper simplifications

### Reject

- `speed * 1.3`
- prompt-cap as a default
- generic Vocos as "real Lux"

### Chosen Current Default

- `zipvoice_distill`
- `4-step`
- Lux/LinaCodec 48k decoder
- Lux solver
- direct transcripts
- `t_shift = 0.9`
- target RMS `0.01`
- no copied speed hack
- no prompt-cap default

---

## Next Concrete Engineering Steps

1. Keep `zipvoice_distill 4-step` as the baseline.
2. Keep the LinaCodec/Lux 48k decoder path as the main infrastructure experiment.
3. Stop spending major time on micro-knob sweeps unless they clearly win.
4. Keep Lux direct model only as an external benchmark, not the base.
5. Move to fine-tuning.
