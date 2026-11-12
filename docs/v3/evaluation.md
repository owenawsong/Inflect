# Inflect v3 evaluation

[Back to Inflect](../../README.md) | [Model scope](release.md)

This page separates intelligibility, automated audio-quality prediction and
runtime speed. None is a substitute for human listening, and no human blind
preference percentages are currently claimed for v3.

## Published Mini: transcription accuracy

| Voice | Equal-dataset average WER |
| :--- | ---: |
| Cassidy | 0.735% |
| Hope | 0.740% |
| Sophia | 0.728% |
| James | 0.741% |
| Jasper | 0.729% |
| Mark | 0.762% |
| **Equal-voice average** | **0.739%** |

Each voice covers 500 CV3 English texts, 1,088 Seed-TTS English texts and 100
MiniMax English texts: **1,688 per voice, 10,128 clips overall**. Scores give
each dataset equal weight, then each voice equal weight. Pooled WER is
**0.875%**, over 127,308 normalized reference words; that is a different
aggregation, not the same statistic.

This is a self-run evaluation using Qwen3-ASR-1.7B and pinned official Open TTS
normalization/alignment code. It is **not an official leaderboard submission**.
The corpus was used during internal development, so this is not a newly held-out
independent test. Hardware, batching and software can differ from the hosted
leaderboard. Private improved candidates are not substituted for the published
six-voice package's score.

[Public protocol](https://huggingface.co/owensong/Inflect-Mini-v3/blob/main/EVALUATION.md)
| [Machine-readable results](https://huggingface.co/owensong/Inflect-Mini-v3/blob/main/evaluation.json)
| [Open TTS Leaderboard](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard)

## Published Mini: automated quality prediction

| Predictor | Mean | Panel |
| :--- | ---: | :--- |
| UTMOSv2 | 3.917 | 180 clips, 30 per voice |
| UTMOS22Strong | 4.479 | The same 180 clips |

These are automated predictors, **not human MOS**. The two systems have different
models and calibrations; do not treat their numerical scales as interchangeable.
These results cover native 24 kHz base speech, not enhancement, reference tuning
or event sounds.

## Current-package warm speed probe

| System | Runtime | RTFx | RTF |
| :--- | :--- | ---: | ---: |
| RTX 5090 / Ryzen 7 7800X3D / Linux | Optimized CUDA, `cuda-source` | **162.1x** | 0.00617 |
| Ryzen 9 3900X / Windows 11 | Canonical CPU, 2 Torch threads | **1.81x** | 0.5519 |

RTFx is audio duration divided by generation time: higher is faster.

The same core weight and package-manifest hashes were verified. Each system
rendered **12 timed clips**: two prompts across six voices, after two excluded
warm-ups. Total generated audio was 98.425 seconds. Timing includes text
normalization, G2P, synthesis and in-memory PCM16 WAV serialization, with CUDA
synchronization around timed work. It excludes loading, verification and disk
writes. Reused text-processing caches can contribute.

GPU: Python 3.12.15, Torch 2.10.0+cu128, 0.607 seconds total warm generation.
CPU: Python 3.12.11, Torch 2.11.0+cpu, 54.317 seconds. Torch used 2 intra-op / 1
inter-op threads on both. The separate load measurements were 3.23 seconds on
GPU and 13.02 seconds on CPU, not full cold process-start measurements.

This is one small warm probe, not a latency distribution, streaming benchmark
or performance guarantee. Different systems and runtime paths mean these rows
are **not a controlled CPU-versus-GPU speedup comparison**.

[Raw runtime results and prompts](https://huggingface.co/owensong/Inflect-Mini-v3/blob/main/assets/runtime-measurements.json)

## Comparing other models

Compare only results with compatible datasets, scoring, voice/reference setup,
output modes and aggregations. Match hardware, software, thread counts, timing
boundaries and warm-up for speed. An official leaderboard's hardware or UTMOS
version must not be silently substituted for the local tests above.

V2's community preference, Modern400 semantic WER and CPU results remain in the
[v2 archive](../v2/README.md). They are not v3 measurements.
