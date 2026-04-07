# Inflect — Full Technical Plan

Last updated: 2026-04-06

---

## Project Goal

Build the world's best open-source TTS under 200M parameters. Features that no one else has at this size:
- Voice cloning
- Actual paralinguistic sounds (real laughs, gasps, etc. — not pitch tricks)
- Emotion control
- Auto-tagging
- Runs locally on CPU

Viral pitch: "Clone any voice. Add [laughs], [gasps], [crying]. Auto-tags your text. 139M params. Runs on CPU. Free forever."

---

## What We ABANDONED and Why

### Kokoro-82M approach (DEAD, DO NOT REVIVE)
- Tried to fine-tune Kokoro to match ElevenLabs mel spectrograms
- HiFiGAN vocoder trained ONLY on clean speech — buzzing/glitching on any paralinguistic sounds
- Even reduced to 24.3% trainable params, audio output was garbage
- Fundamental architecture mismatch: HiFiGAN cannot reproduce ElevenLabs-style mels
- Lesson: Cannot add paralinguistic sounds to a model whose vocoder has never seen them

---

## Base Model: Pocket TTS (100M)

**Why Pocket TTS:**
- Sounds significantly more human/emotional than Kokoro (which is robotic)
- 100M params, CPU-optimized, streaming
- Voice cloning built-in: `tts_model.get_state_for_audio_prompt("voice.wav")`
- Uses Mimi VAE codec (32-dim continuous latents, 12.5 Hz)
- FlowLM: 6-layer transformer, 1024 d_model, flow matching with LSD
- CC-BY-4.0 license (open, can build on top)
- Trained on 88k hours of audio

**Architecture internals (critical for our work):**
- Mimi encoder: SEANet, downsampling [6,5,4], hop_length=120, 24kHz
- Mimi latent: 32-dim continuous, 12.5 Hz (1 frame = 80ms)
- Speaker projection: `speaker_proj_weight` (1024×512) — maps Mimi speaker latent to conditioning
- FlowLM generates latents via LSD (1 step for speed)
- No training code publicly available — BUT EmoShift works by injecting into output latents

**Key constraint:** No Pocket TTS training code. Anything we do must either:
1. Work as a separate module generating/modifying Mimi latents
2. Use the EmoShift method (inject steering vectors into output)

---

## Full System Architecture

```
Text Input: "That's so funny [laughs] I can't believe it [emotion: excited]"
                     ↓
              [AUTO-TAGGER] (optional toggle)
              Adds tags to plain text if missing
                     ↓
              [TEXT PARSER]
              Splits into: speech segments, event tags, emotion context
              → ["That's so funny", [laughs], "I can't believe it"]
              → emotion: excited
                     ↓
         ┌───────────────────────────────┐
         │     Parallel Processing       │
         │  [POCKET TTS]  [PARA MODULE]  │
         │  Speech → Mimi  Tag → Mimi    │
         │    latents       latents      │
         └───────────────────────────────┘
                     ↓
              Concatenate Mimi latents in order
                     ↓
              [EMOTION STEERING]
              Add emotion steering vector to all latents (EmoShift)
                     ↓
              [VOICE STYLE ADAPTER]
              GE2E embedding refines Mimi latents (speaker texture/variation)
                     ↓
              [SINGLE MIMI DECODER PASS]
              One decode → no splicing artifacts
                     ↓
                  Audio Output
```

---

## Component Details

### 1. Pocket TTS (100M) — FROZEN, NEVER TOUCH WEIGHTS

Use exactly as-is for voice cloning + speech generation.
Voice cloning: pass raw audio file → get Mimi speaker state → condition generation.
Our GE2E encoder enriches this in the Voice Style Adapter step.

---

### 2. Paralinguistic Module (23M) — THE CORE INNOVATION

**Architecture:**
```python
class ParaModule(nn.Module):
    # Input conditioning
    speaker_mimi_proj: Linear(32 * N_frames, 128)   # compress speaker state
    tag_embedding: Embedding(32_tags, 32)             # which sound?
    emotion_proj: Linear(16, 32)                      # emotion context
    intensity_proj: Linear(1, 16)                     # how intense?

    # Small transformer decoder
    transformer: TransformerDecoder(
        d_model=256, nhead=4, num_layers=4
    )
    duration_pred: Linear(256, 1)   # predict output length
    output_proj: Linear(256, 32)    # → Mimi latent dims
```

**Supported event tags:**
```
Laughter:   [laughs], [laughs_hard], [chuckles], [giggles]
Breath:     [sighs], [sighs_deeply], [exhales], [inhales], [gasps]
Sadness:    [crying], [whimpers], [sobs], [sniffles]
Filler:     [hmm], [um], [uh], [breath]
Throat:     [clears_throat], [coughs]
Silence:    [pause Xs] (X = seconds), [silence Xs]
```

**Why Mimi latent space is key:**
- Speech latents from Pocket TTS: 32-dim × N frames
- Para Module output: also 32-dim × M frames
- Concatenate → single Mimi decode → ZERO splicing artifacts
- The laugh/gasp sounds are generated in the same representational space as speech

**Training pipeline:**
```
Step 1: SenseVoice ASR on our 1070 ElevenLabs clips
        → transcripts with [tag] position timestamps

Step 2: Snip the paralinguistic segments from each clip
        (e.g., extract just the laugh portion at the [laughs] timestamp)

Step 3: SeedVC zero-shot voice conversion
        → convert each snipped segment to all 7 voice styles
        → 7× more training data, voice-matched

Step 4: Extract Mimi latents from all voice-converted segments
        → Training pairs: (speaker_state, tag_id, target_mimi_latents)

Step 5: Train Para Module
        - Phase 1 SFT: L1(predicted_latents, target_latents), 50 epochs
        - Phase 2 DPO: reject=no sound, choose=para sound, β=0.01

Cost: ~$3 CAD on RTX 3090 via Vast.ai (~2-3 hours)
```

---

### 3. Emotion Steering (0 new params) — EmoShift Method

**Confirmed approach:** A Reddit user confirmed that EmoShift works with Pocket TTS.
They used CREMA-D dataset, trained steering vectors per emotion, inject into Pocket output during inference.
The results were described as "emotions sounding similar" — we need to do it better.

**Our improved approach:**
```python
# One-time setup:
for emotion in ['sad', 'excited', 'angry', 'happy', 'fearful', 'disgusted', 'surprised']:
    examples = load_emotion_examples(emotion)   # From CREMA-D or our ElevenLabs data
    latents = [run_pocket_tts(ex) for ex in examples]
    neutral_latents = [run_pocket_tts(neutral_ex) for neutral_ex in neutral_examples]
    steering_vectors[emotion] = mean(latents) - mean(neutral_latents)

# At inference:
generated_latents += alpha * steering_vectors[emotion]   # alpha = intensity [0.0, 1.0]
```

**Why we'll do better than the Reddit user:**
1. Use multiple steering layers, not just one
2. Test PCA on the steering directions for more orthogonal emotion axes
3. Use our ElevenLabs emotional data as training signal (7 voices × multiple emotions)
4. Fine-tune injection depth (which layers/where to inject)

**Emotion tags:**
```
[emotion: sad]       [emotion: excited]    [emotion: angry]
[emotion: happy]     [emotion: fearful]    [emotion: disgusted]  [emotion: surprised]
```

---

### 4. Voice Style Adapter (5M) — Better Voice Cloning

**Problem:** Pocket TTS voice cloning gets rough speaker identity (pitch, rhythm) but misses:
- Vocal texture
- Speaking rate variation (some people are very variable, others monotone)
- Breathiness / vocal fry
- Emotional range of the speaker

**Solution:**
```python
class VoiceStyleAdapter(nn.Module):
    # GE2E encoder (existing, 11.3M, FROZEN) → 256-dim speaker embedding
    # This captures the fine-grained speaker characteristics

    style_proj: Linear(256, 512)
    adapter_layers: TransformerEncoder(d_model=32, nhead=4, num_layers=3)
    # Takes: Mimi latents from Pocket + style projection
    # Output: Refined Mimi latents (same shape, style-corrected)

# Training:
# Triplet loss: same_speaker_refined closer to ground truth than diff_speaker
# Data: our 7 voices × 1070 clips
# Cost: ~$2 CAD
```

**GE2E encoder we already have:**
- `voice-encoder/checkpoints/1d_ref_epoch_0500.pt` (most recent)
- 11.3M params, trained on our 7-voice dataset
- Outputs 256-dim speaker embedding
- Already handles voice similarity well

---

### 5. Auto-Tagger — The UX Feature

**Mode 1: Rule-based (always on, instant)**
```python
rules = {
    r'\blol\b|\bhaha\b|\blmao\b'    → '[laughs]',
    r'\.\.\.'                         → '[sighs]',
    r'omg|oh my god|no way!'         → '[gasps]',
    r'\*whispers?\*|sssh|shh'        → '[whispers]',
    # typo corrections too
}
```

**Mode 2: Semantic classifier (optional, ~3M)**
- Fine-tune DistilBERT for text→tag prediction
- More accurate on nuanced cases
- Can be disabled if latency matters

**Toggle:** `inflect.set_autotagger(enabled=True, mode='rule')` or `mode='semantic'`

---

### 6. Custom Voice Fine-Tuning (LoRA, 3M per user)

**Goal:** User uploads 1-2 minutes of their own voice → personalized Pocket TTS model

**Method:** Since we can't access Pocket TTS internals for full fine-tuning, we:
1. Extract Mimi latents from user audio
2. Train a small LoRA adapter on the speaker_proj_weight layer
3. Save adapter (3M params, tiny file)
4. At inference: load LoRA on top of frozen Pocket TTS

**Alternative:** Since EmoShift training code can be shared by the Reddit OP (they offered),
we may be able to extend this approach to also fine-tune speaker conditioning.

---

### 7. Voice Blending (0 new params)

```python
def blend_voices(voice_a_state, voice_b_state, alpha=0.5):
    # Linear interpolation of Mimi speaker states
    return alpha * voice_a_state + (1 - alpha) * voice_b_state

# Usage:
state = blend_voices(morgan_freeman_state, scarlett_johansson_state, alpha=0.7)
```

Simple, works immediately, great for demos.

---

### 8. Pause/Silence Control (Must-Have, trivial)

```python
# At text parsing stage:
[pause 0.5s]  → insert 0.5 * 12.5 = ~6 zero frames into Mimi latents
[silence 1.0s] → insert 12 zero frames
[breath]      → insert pre-computed breath Mimi latents (from Para Module)
```

---

## Tag Syntax (Final Design)

```
EVENT TAGS (inline, happen at a point):
  Sounds:   [laughs] [laughs_hard] [chuckles] [giggles]
            [gasps] [sighs] [sighs_deeply] [exhales] [inhales]
            [crying] [whimpers] [sobs] [sniffles]
            [coughs] [clears_throat] [hmm] [um] [uh] [breath]
  Timing:   [pause 0.5s] [silence 1.0s]

EMOTION TAGS (global, affect generation style):
  [emotion: sad] [emotion: excited] [emotion: angry] [emotion: happy]
  [emotion: fearful] [emotion: disgusted] [emotion: surprised]

PROSODY SPANS (LATER, lower priority):
  <slow>text</slow>   <fast>text</fast>
  <soft>text</soft>   <hard>text</hard>
```

---

## Repository Structure (Target)

```
Inflect/
├── README.md
├── CLAUDE_READ/             ← AI reference files, not shipped to users
│   ├── claude.md            ← My notes/head
│   ├── plan.md              ← This file
│   ├── todo.md              ← Current todos
│   └── architecture.md     ← Diagrams / deep dives
│
├── pocket-tts-pipeline/     ← NEW main pipeline
│   ├── inflect.py           ← Main API (single entry point)
│   ├── para_module/         ← Paralinguistic Module
│   │   ├── model.py
│   │   ├── train.py
│   │   └── tags.py
│   ├── emotion/
│   │   ├── steering.py      ← EmoShift implementation
│   │   └── vectors/         ← Saved .pt steering vectors
│   ├── auto_tagger/
│   │   ├── rules.py
│   │   └── classifier.py
│   ├── voice_adapter/
│   │   ├── model.py
│   │   └── train.py
│   └── data/
│       ├── prepare.py       ← SenseVoice + SeedVC pipeline
│       └── extract_mimi.py
│
├── voice-encoder/           ← KEEP (GE2E encoder, trained checkpoints)
├── emotion-deltas/          ← KEEP (might be useful as reference)
└── kokoro-finetune/         ← ARCHIVE (old approach, abandoned)
```

---

## Budget

| Task | Cost | GPU |
|---|---|---|
| Data pipeline (local) | $0 | Local |
| Para Module SFT (2hr) | ~$1.50 CAD | RTX 3090 @ $0.15/hr |
| Para Module DPO (1hr) | ~$0.75 CAD | RTX 3090 |
| Voice Style Adapter (1hr) | ~$0.75 CAD | RTX 3090 |
| Reruns / experiments | ~$3 CAD | RTX 3090 |
| **Total** | **~$6 CAD** | |
| Remaining budget | ~$24 CAD | For demos / polish |

---

## Success Metrics

- [laughs] generates an actual audible laugh in the cloned voice
- [emotion: sad] noticeably changes prosody/delivery
- Auto-tagger correctly tags "lol" → [laughs] and "..." → [sighs]
- Voice cloning preserves identity across tags
- Full pipeline runs under 1 second (streaming) on CPU
- HuggingFace demo works in browser
