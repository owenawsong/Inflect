# Inflect — Execution Roadmap

Last updated: 2026-04-13

## Project Goal

Build a stronger English zero-shot voice-cloning stack by:

1. choosing a stable ZipVoice-based runtime
2. building a high-value synthetic foundation dataset
3. fine-tuning the teacher cleanly
4. adding expressive range only after the foundation improves
5. training the enhancer last

## Current Position

The runtime decision is mostly done.

Chosen runtime:

- `inflect_base + lux solver`

The main blockers now are:

- scaling the dataset cleanly
- evaluating the teacher fine-tune properly
- cleaning up the public repo and release story

## Phase 1 — Runtime Foundation

Status: mostly complete

Completed:

- official ZipVoice baselines run locally
- real LuxTTS inspected and benchmarked
- Lux speed hack rejected
- Lux solver and decoder ideas isolated
- current runtime default chosen

Remaining optional runtime work:

- `k2`-enabled benchmark sanity check
- reference-voice leakage audit
- runtime polish only if a clear regression appears

## Phase 2 — VoxCPM Dataset Scaling

Status: active

Current assets:

- `20,000`-text English `v2` corpus
- `60,000`-clip balanced generation plan
- `56` reference voices

Current active dataset run:

- `outputs/voxcpm_dataset/20260411_large_text_v1`

Current public-release approach:

- keep the active run for immediate work
- later freeze a clean public snapshot

## Phase 3 — Teacher Fine-Tuning

Status: active

Current direction:

- fine-tune the ZipVoice teacher first
- keep the chosen fast runtime stack for inference

Current state:

- local training loop now runs again
- non-finite crash fixed through the local ZipVoice patch
- demo and smoke runs are now valid again

Immediate goal:

- finish a short demo run
- listen and compare
- if promising, start the first proper foundation run

## Phase 4 — Expressive Enrichment

Status: not started

Planned after the foundation pass:

- add Expresso or similar expressive data
- improve prosody and emotional richness
- avoid mixing too many datasets before the foundation signal is clear

## Phase 5 — Inflect-Enhance

Status: implemented but parked

Rule:

- do not train it fully until the backbone and its artifact profile are stable

## Phase 6 — Publishing

Status: active cleanup

Goals:

- make the GitHub repo understandable
- separate source from local artifacts
- publish a real Hugging Face dataset snapshot
- document licensing and provenance honestly

## Short-Term Execution Order

1. keep VoxCPM generation running
2. finish the short teacher demo
3. update GitHub docs and structure
4. freeze and publish a public dataset snapshot
5. run the first longer foundation fine-tune
6. evaluate before adding expressive data
