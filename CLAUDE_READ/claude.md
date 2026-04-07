# Claude's Head — Inflect Project Notes

This file is my personal reference. I read this when I lose track of context.
Last updated: 2026-04-06

---

## CRITICAL: What We're Building

Inflect = Pocket TTS (100M, frozen) + Paralinguistic Module (23M) + Emotion Steering + Voice Adapter

We are NOT fine-tuning Pocket TTS weights. We work AROUND them in Mimi latent space.

---

## What We ABANDONED

**Kokoro fine-tuning is DEAD. Do not suggest reviving it.**
- HiFiGAN vocoder = trained on clean speech only = buzzes on paralinguistic mels
- Even 24.3% trainable params produced garbage audio
- The architecture is fundamentally wrong for this use case

---

## Owen's Preferences (Critical)

From feedback_style.md:
- Direct, no filler, no fluff
- Edit files directly
- Fast-paced
- Owen is a student/dev in Canada, RTX 3060 12GB locally, ~$30 CAD total GPU budget

What Owen does NOT want:
- Being asked questions when I should just decide
- Overexplaining obvious things
- Slow, robotic-sounding TTS (he thinks Kokoro sounds like a robot)
- Options when he wants a recommendation

What Owen DOES want:
- Pocket TTS (sounds way more human than Kokoro)
- Real paralinguistic sounds (actual laughing, not pitch tricks)
- Something that can go viral on X, Reddit, GitHub, HuggingFace
- Model under 200M params total
- Under $30 CAD total GPU cost

---

## Key Technical Facts

### Pocket TTS
- 100M params, Mimi VAE codec, FlowLM (flow matching, not diffusion)
- Mimi latent: 32-dim continuous, 12.5 Hz, 24kHz audio
- Voice cloning: `tts_model.get_state_for_audio_prompt("voice.wav")`
- NO public training code — but EmoShift injection into output latents WORKS (confirmed on Reddit)
- License: CC-BY-4.0

### EmoShift (confirmed method for Pocket TTS)
- A Reddit user (r/LocalLLaMA post) fine-tuned Pocket TTS emotion using this
- Method: small steering vector per emotion, injected into model output
- Used CREMA-D dataset for training
- Their result: emotions "sound the same" (not good) — we need to do better
- They offered to share training code — worth reaching out if we need it

### Our Existing Assets
- voice-encoder/checkpoints/1d_ref_epoch_0500.pt = best GE2E encoder (11.3M params)
- emotion-deltas/deltas.pt and deltas_v2.pt = old Kokoro emotion steering (may be useful as reference)
- voice-encoder/data/ = 1070 ElevenLabs clips across 7 voices with paralinguistic tags
- voice-encoder/data/manifest.csv = the data index
- 7 voices: Amy, Arabella, Austin, Bradford, Charlotte, Hope, James

### Training Data
- 1070 clips, 7 voices (ElevenLabs)
- Each clip: has a [tag] in text, audio contains that sound
- Tags include: [laughs], [sighs], [whispers], [gasps], [crying], etc.
- Already phonemized (from old Kokoro work) but we don't need phonemes for Pocket TTS

---

## Decisions Made (Do Not Re-Debate)

1. **Base model = Pocket TTS.** Not Kokoro. Not Dia. Not Orpheus. Pocket TTS.
2. **Work in Mimi latent space** for all additions (Para Module outputs Mimi latents).
3. **No direct Pocket TTS weight modification** (no training code available).
4. **Tag syntax:** `[event_tags]` for sounds, `[emotion: X]` for global emotion, `<modifier>text</modifier>` for prosody spans (later).
5. **Breathing patterns = deprioritized** (might sound artificial).
6. **Speaking styles = just emotion steering** (not a separate feature).
7. **Background ambience = deprioritize**.
8. **Gender/age shifting = just use different voices**.
9. **Pause/silence = MUST HAVE**, not nice-to-have.

---

## Current Project State

- [x] Old Kokoro fine-tuning code exists but is abandoned
- [x] GE2E voice encoder trained (1d_ref_epoch_0500.pt = best)
- [x] 1070 ElevenLabs clips collected
- [x] Old emotion deltas computed (for Kokoro, not Pocket TTS)
- [ ] Pocket TTS pipeline not yet set up
- [ ] Para Module not written
- [ ] SenseVoice + SeedVC data pipeline not built
- [ ] Emotion steering not implemented for Pocket TTS
- [ ] Voice Style Adapter not written
- [ ] Auto-tagger not written

**Next steps: see todo.md**

---

## File Locations

```
/c/Users/Owen/Inflect-New/            ← Main repo
/c/Users/Owen/Inflect-New/voice-encoder/data/  ← 1070 training clips
/c/Users/Owen/Inflect-New/voice-encoder/checkpoints/  ← GE2E checkpoints
/c/Users/Owen/Inflect-New/CLAUDE_READ/  ← This folder (my reference)
```

GitHub: https://github.com/owenawsong/Inflect

---

## Things To Remember

- Always check todo.md at the start of a new session
- Update this file and todo.md after major decisions or completions
- The Para Module must output Mimi latents (32-dim × T frames, 12.5 Hz) — not mel spectrograms, not raw audio
- SeedVC is the voice conversion tool for the data pipeline
- SenseVoice is the ASR tool for detecting paralinguistic timestamps
- DPO training is better than SFT for paralinguistic quality (proven in SynParaSpeech paper)
- The training must happen on cloud GPU (Owen has ~$24 remaining of $30 budget)
