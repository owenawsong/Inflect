# Inflect Roadmap

Inflect is being developed as a small, local-first English TTS system with a strong emphasis on stability, emotion, and practical voice cloning.

## North Star

Build a compact speech model that feels alive:

- short-reference zero-shot voice cloning
- natural pacing and pauses
- reliable long-prompt generation
- expressive delivery without random emotional swings
- optional enhancement for final audio polish
- local inference on consumer hardware

## Release Tracks

### Track 1: Stabilize the TTS Core

Current work focuses on the Lux / ZipVoice lineage because it has the best tiny-model expressiveness we have heard so far.

Open problems:

- inconsistent speaking speed
- long-prompt fragility
- occasional glitches or mumbles
- weak internal pause control
- fine-tuning instability when too much of the model is updated

Active strategy:

- prefer native runtime paths over wrappers when possible
- use tiny damage tests before long fine-tunes
- compare checkpoints through blind A/B
- add objective checks for duration, silence, RMS jumps, and text coverage

### Track 2: Runtime Stabilizer

Some failures should be caught at inference time instead of hidden inside a checkpoint.

Planned runtime guardrails:

- sentence-aware chunking for long prompts
- target duration ratio checks
- leading-click trimming and short edge fades
- silence and RMS anomaly detection
- retry on obvious bad generations
- crossfade stitching for multi-chunk output

### Track 3: Better Training Data

The model should learn stable speech first, then emotion and tags.

Data stages:

1. high-quality stable English speech
2. voice-cloning diversity
3. expressive speech
4. paralinguistic tags
5. enhancer pairs

Synthetic data is useful, but it must be filtered. A model that learns teacher artifacts will sound worse even if the loss improves.

### Track 4: Inflect-Enhance

Inflect-Enhance is an optional post-generation restoration model.

It should improve:

- detail
- perceived bandwidth
- hiss or metallic artifacts
- small codec/vocoder defects

It should not be used to hide:

- skipped words
- bad pacing
- wrong emotion
- hallucinated speech
- broken voice cloning

### Track 5: Explicit Style and Paralinguistic Control

Long-term goal:

```text
<soft> I did not expect that. <breath> Are you sure?
<laugh> No, no, that actually worked.
<whisper> Keep your voice down.
```

The first release does not need every tag. It needs the architecture and data path to support tags cleanly.

## Milestones

| Milestone | Goal | Exit Criteria |
| --- | --- | --- |
| M0 | Public repo scaffold | README, docs, eval plan, clean publish rules. |
| M1 | Runtime benchmark suite | Repeatable blind A/B and objective metrics. |
| M2 | Stable Lux/ZipVoice candidate | No regressions vs base on stress prompts. |
| M3 | Inflect-Nano preview | Usable local voice cloning checkpoint. |
| M4 | Sample gallery | Honest demos with wins and failures. |
| M5 | Inflect-Enhance preview | Optional small enhancer with before/after examples. |
| M6 | Tagged expressiveness | First controllable style/paralinguistic subset. |

## Non-Goals For The First Release

- multilingual support
- celebrity voice cloning
- giant hosted-only inference
- pretending synthetic data has no provenance issues
- claiming a tiny parameter count for a larger full pipeline

## Current Decision Rule

If a checkpoint improves one clip but breaks long prompts, it is not progress.

If a model sounds great only after cherry-picking, it is not release-ready.

If the enhanced pipeline is used in demos, the enhancer must be disclosed as part of the pipeline.
