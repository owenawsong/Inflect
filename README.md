# Inflect

A local, open-source TTS system with voice cloning, real paralinguistic sounds, and emotion control — under 200M parameters.

> **Status: Active development. No release yet.**

---

## What It Does

- **Voice Cloning** — clone any voice from a short audio clip
- **Paralinguistic Sounds** — actual `[laughs]`, `[gasps]`, `[sighs]`, `[crying]`, not pitch-shifted phonemes
- **Emotion Control** — `[emotion: sad]`, `[emotion: excited]`, etc.
- **Auto-Tagger** — toggle on to automatically add tags from plain text
- **Pause/Silence Control** — `[pause 0.5s]`, `[silence 1.0s]`
- **Custom Voice Fine-Tuning** — upload your voice, get a personalized model
- **Voice Blending** — mix two voices together

---

## Architecture

| Component | Params | Role |
|---|---|---|
| Pocket TTS (frozen) | 100M | Core speech synthesis, voice cloning |
| Paralinguistic Module | ~23M | Generates event sounds in Mimi latent space |
| Voice Style Adapter | ~5M | Enriches voice cloning with GE2E detail |
| Emotion Steering | ~0M | EmoShift vectors injected at inference |
| Auto-Tagger | ~0M | Rule-based + optional classifier |
| **Total** | **~139M** | |

---

## Tag Syntax

```
Event tags (inline, one-time sounds):
  [laughs]  [laughs_hard]  [chuckles]  [gasps]  [sighs]  [sighs_deeply]
  [crying]  [whimpers]  [sobs]  [coughs]  [clears_throat]  [hmm]  [um]
  [pause 0.5s]  [silence 1.0s]  [breath]

Emotion tags (global, affects generation style):
  [emotion: sad]  [emotion: excited]  [emotion: angry]  [emotion: happy]
  [emotion: fearful]  [emotion: disgusted]  [emotion: surprised]
```

---

## Why Under 200M?

ElevenLabs is proprietary and paid. Orpheus is 3B with no voice cloning. Dia is 1.6B. Inflect does everything at 139M and runs fully locally on CPU.

---

## License

Pocket TTS base: CC-BY-4.0 (Kyutai)
Inflect additions: Apache 2.0
