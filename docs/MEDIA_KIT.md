# Media Kit

This folder is the staging area for public-facing launch assets.

## Planned Assets

| Asset | Status | Notes |
| --- | --- | --- |
| Hero wordmark | Placeholder | [`assets/inflect-wordmark.svg`](../assets/inflect-wordmark.svg) |
| Demo video card | Placeholder | [`assets/demo-video-placeholder.svg`](../assets/demo-video-placeholder.svg) |
| Sample gallery card | Placeholder | [`assets/sample-card-placeholder.svg`](../assets/sample-card-placeholder.svg) |
| 30-45s demo video | Planned | Voice cloning, emotion, local inference, enhancer. |
| Audio sample gallery | Planned | Base vs tuned vs enhanced. |
| Architecture diagram | Draft | README Mermaid diagram. |

## Demo Video Structure

Target length: 30-45 seconds.

Suggested arc:

1. **Hook:** "A small local voice model should not sound small."
2. **Voice clone:** show short reference audio becoming generated speech.
3. **Emotion:** same text with restrained, excited, and soft delivery.
4. **Long prompt:** prove it does not collapse under real text.
5. **Enhance:** raw vs enhanced detail.
6. **Close:** "Inflect-Nano. Expressive speech, locally."

## Visual Direction

Use a warm editorial style rather than generic futuristic AI visuals.

Direction:

- dark warm background
- cream text
- amber/orange waveform accents
- restrained motion
- premium audio-tool feeling
- no neon sci-fi UI unless the product itself needs it

## Sample Gallery Template

Each sample should list:

- voice
- prompt
- model variant
- speed/steps/runtime settings
- whether enhancer is used
- known issues

Recommended comparison:

```text
Base
Tuned
Tuned + Inflect-Enhance
```

## Honesty Rule

If a sample uses an enhancer, label it.

If a sample is cherry-picked, do not present it as representative.

If a model fails on long prompts, show the fix or state the limitation.
