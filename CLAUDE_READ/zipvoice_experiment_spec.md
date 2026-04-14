# ZipVoice Experiment Spec

Last updated: 2026-04-12

---

## Purpose

This file is the concrete Phase 1 and Phase 2 execution spec.

The goal is not vague "improvement."
The goal is:
- build a repeatable benchmark harness
- compare official ZipVoice variants first
- then test Lux-inspired changes one by one

---

## Benchmark Voice Set

Start with a tight subset of 6 voices:
- 3 female
- 3 male

Selection rules:
- clean audio
- solid transcripts
- not obviously noisy or damaged
- not all the same accent / tone

Why only 6 first:
- enough diversity to expose instability
- small enough to test quickly and repeatedly

---

## Benchmark Prompt Set

Use 8 prompts total.

Required buckets:
1. very short neutral
2. short conversational
3. medium conversational
4. long narrative
5. punctuation / rhythm stress
6. emotional intensity
7. disbelief / frustration
8. clone-stress sentence with difficult pacing

Rules:
- prompts must be fixed once selected
- do not keep changing them between variants

---

## Baseline Variants To Run First

Run these before any custom changes:

1. `zipvoice`
- official base model
- official default step count

2. `zipvoice_distill_8step`
- official distill path
- 8 steps

3. `zipvoice_distill_4step`
- distill fast path
- 4 steps

This is the first real decision point:
- if `distill_4step` is too degraded, it fails
- if `distill_8step` is still too clipped, it fails
- if base `zipvoice` is clearly best, it becomes the control

---

## Output Layout

```text
outputs/zipvoice_bench/
  zipvoice/
    metadata.json
    <voice_name>/
      01_short_neutral.wav
      02_short_conversational.wav
      03_medium_conversational.wav
      04_long_narrative.wav
      05_punctuation.wav
      06_emotional.wav
      07_disbelief.wav
      08_clone_stress.wav
  zipvoice_distill_8step/
  zipvoice_distill_4step/
  variant_<name>/
```

---

## Metadata To Record

For each variant:
- model name
- step count
- guidance scale
- speed
- t-shift
- prompt RMS
- reference voice list
- prompt list
- total runtime
- generation failures

For each sample during listening:
- clarity: 1 to 5
- cloning similarity: 1 to 5
- natural pacing: 1 to 5
- cut-off severity: 0 to 3
- glitch severity: 0 to 3
- buzz / noise severity: 0 to 3

---

## Phase 2 Test Order

Only after the three baseline variants above are saved.

### Test A — Prompt preprocessing
- prompt duration cap
- prompt RMS choices
- maybe trimmed leading/trailing silence

### Test B — Distill inference settings
- 4-step versus 6-step versus 8-step if worth it
- only on the distill branch

### Test C — Solver / sampler changes
- inspect Lux for the actual delta
- port only the solver change first
- no vocoder changes in the same experiment

### Test D — Vocoder path changes
- only if they can be implemented correctly
- reject immediately if timing or duration breaks

---

## Promotion Criteria

A change survives only if:
- average clarity improves
- cloning does not regress
- cut-off rate does not get worse
- timing remains sane
- results stay consistent across multiple voices

Reject any variant that:
- sounds great on one clip but unstable on others
- creates speed / duration mismatches
- introduces new buzz or glitches

---

## Immediate Next Deliverable

Build:
- `scripts/test_base_zipvoice.py`

It should:
- run the fixed 6 voices × 8 prompts grid
- support `zipvoice` and `zipvoice_distill`
- support step overrides
- write organized outputs plus metadata

That is the first script that matters now.
