# Inflect — Todo List

Last updated: 2026-04-06

---

## Phase 1: Data Pipeline (Local, $0)

- [ ] Set up Pocket TTS locally and confirm inference works
- [ ] Run SenseVoice on 1070 clips → get transcripts + paralinguistic timestamps
- [ ] Snip paralinguistic segments from each clip using timestamps
- [ ] Run SeedVC voice conversion on snipped segments (→ 7 voices each)
- [ ] Extract Mimi latents from all voice-converted segments
- [ ] Build training dataset: (speaker_state, tag_id, target_mimi_latents) pairs
- [ ] Verify dataset quality (listen to samples)

## Phase 2: Paralinguistic Module (~$3 CAD)

- [ ] Write Para Module architecture (model.py)
- [ ] Write training script (train.py) — SFT first, then DPO
- [ ] Upload dataset to Vast.ai
- [ ] Train SFT phase (~2 hours, RTX 3090)
- [ ] Train DPO phase (~1 hour, RTX 3090)
- [ ] Test: does [laughs] produce an actual laugh sound?
- [ ] Test: does [sighs] produce an actual sigh?

## Phase 3: Emotion Steering ($0)

- [ ] Download CREMA-D dataset
- [ ] Run Pocket TTS on CREMA-D examples per emotion class
- [ ] Extract Mimi latents per class
- [ ] Compute steering vectors (mean per emotion - mean neutral)
- [ ] Test EmoShift injection at inference
- [ ] Tune injection alpha for natural-sounding results
- [ ] Save emotion vectors as pocket-tts-pipeline/emotion/vectors/*.pt

## Phase 4: Voice Style Adapter (~$2 CAD)

- [ ] Write Voice Style Adapter architecture
- [ ] Write training script (triplet loss, speaker similarity)
- [ ] Train on our 7 voices × 1070 clips
- [ ] Test: does the adapter improve voice texture vs. Pocket native cloning?

## Phase 5: Auto-Tagger ($0)

- [ ] Write rule-based tagger (rules.py)
- [ ] Test on common cases: "lol" → [laughs], "..." → [sighs]
- [ ] Add typo correction rules
- [ ] (Optional later) Fine-tune DistilBERT classifier for semantic mode

## Phase 6: Integration

- [ ] Write main inflect.py API
- [ ] Full pipeline: parse → generate → stitch Mimi latents → decode
- [ ] Implement [pause X] and [silence X] as zero-frame insertion
- [ ] Test full pipeline end-to-end
- [ ] Benchmark: latency on CPU, latency on GPU
- [ ] Fix any streaming issues

## Phase 7: Demo + Release

- [ ] Record demo audio: plain text vs. auto-tagged vs. manual tags
- [ ] Comparison audio: Inflect vs. ElevenLabs vs. Kokoro vs. Pocket TTS vanilla
- [ ] Set up HuggingFace Space (Gradio demo)
- [ ] Write model card
- [ ] Push to GitHub with proper license
- [ ] Post to r/LocalLLaMA, r/MachineLearning, X, HuggingFace

---

## Parking Lot (Later, Not Now)

- [ ] Prosody spans: `<slow>text</slow>`, `<soft>text</soft>` — complex, do AFTER core features
- [ ] Custom Voice Fine-Tuning (LoRA) — nice to have, do after everything else works
- [ ] Voice Blending (trivial once pipeline is set up)
- [ ] Multi-language support (needs retraining Pocket TTS, not in scope)
- [ ] WebGPU / browser demo (possible since Pocket is CPU-optimized, do last)
- [ ] Contact Reddit OP about their EmoShift training code
