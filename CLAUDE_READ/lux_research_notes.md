# LuxTTS Research Notes

Last updated: 2026-04-12

---

## What LuxTTS Actually Changes

After inspecting the local Lux code against official ZipVoice, the real differences are:

1. `ZipVoiceDistill` base
- Lux uses the distilled model branch, not base `zipvoice`
- `third_party/luxtts_clean/zipvoice/models/zipvoice_distill.py`
- no architectural differences from the official distilled model file were found

2. Custom solver update rule
- This is the first real code-level inference change that matters
- official ZipVoice uses plain Euler-style update:
  - `x = x + v * dt`
- Lux solver changes the update into an anchor-style probability-flow step:
  - predict `x_1` and `x_0`
  - advance by anchoring to the line between them
  - snap final step directly to `x_1_pred`
- File:
  - `third_party/luxtts_clean/zipvoice/models/modules/solver.py`

3. Custom 48kHz vocoder path
- Lux does not just use stock 24k Vocos
- it expects custom vocoder components tied to LinaCodec
- concrete decoder structure:
  - normal 24k branch
  - `upsampler`
  - `head_48k`
  - Linkwitz-Riley crossover merge between resampled 24k branch and 48k branch
- this is the main reason our local fallback did not reproduce true Lux behavior
- this is also the strongest candidate for the audible Lux improvement

4. Wrapper-level inference heuristics
- wrapper defaults to 4-step generation
- wrapper exposes `rms`, `t_shift`, `speed`, `return_smooth`
- local code also applies `speed = speed * 1.3` inside generation
- prompt audio is loaded with a duration cap using librosa

Assessment:
- `speed * 1.3` should not be copied as an Inflect default
- prompt duration caps are too risky to make default

5. Packaging / runtime streamlining
- simple wrapper class
- optional Whisper transcription
- ONNX/CPU path bundled

---

## What LuxTTS Does NOT Prove

It does not prove there is a deep architecture rewrite.

The strongest observed improvements come from:
- distilled path
- solver change
- vocoder change
- runtime heuristics

But not all heuristics are good:
- decoder path: likely good
- speed multiplier: bad default
- aggressive prompt cap: risky

So the correct strategy is:
- port these one by one
- benchmark each against official ZipVoice

---

## Ranked Experiment Order

### Tier 1 — highest signal

1. `zipvoice_distill` 4-step baseline
2. LinaCodec / Lux 48k decoder path
3. Track A tuned stack without copied speed hacks

### Tier 2 — medium signal

4. Lux-style anchored solver on top of official `zipvoice_distill`
5. prompt RMS experiments
6. raw-vs-processed prompt path comparisons

### Tier 3 — low signal / cleanup

7. `t_shift` sweeps
8. prompt-duration-cap sweeps

Those are not the main lever anymore.

---

## Current Decision

Working baseline:
- official `zipvoice_distill`
- 4-step

Current strongest Lux-side component:
- LinaCodec / Lux 48k decoder path

Current rejected Lux default:
- `speed * 1.3`
