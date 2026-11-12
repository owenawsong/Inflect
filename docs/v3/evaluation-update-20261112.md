# Mini v3: Cassidy ASR Recheck (2026-11-12)

This fresh **Cassidy-only** check uses the published default Mini synthesis
weights. It introduces no model-weight change and does **not** replace the
historical **0.739% six-voice WER** over 10,128 clips in the
[published evaluation](evaluation.md).

## Result

Native 24 kHz, mono PCM16, seed 42, default/base speech from raw text. No optional
delivery profile, vocal events, voice tuning or 48 kHz enhancement was enabled.
WER is a percentage; lower is better.

| Corpus | Clips | Reference words | S / D / I | Edits | WER % |
| --- | ---: | ---: | --- | ---: | ---: |
| CV3 English | 500 | 7,377 | 70 / 5 / 6 | 81 | 1.09800732 |
| Seed-TTS English | 1,088 | 11,941 | 83 / 10 / 2 | 95 | 0.79557826 |
| MiniMax English | 100 | 1,900 | 4 / 1 / 0 | 5 | 0.26315789 |

**Equal-dataset macro WER: 0.71891449%.** This is the arithmetic mean of
the three dataset WERs, not a per-clip mean or a new six-voice average.
**Pooled WER, separately: 0.85304930%**, from 181 edits / 21,218 reference words
across all 1,688 clips (157 substitutions, 16 deletions, 8 insertions).

## Local Protocol And Limits

Qwen3-ASR-1.7B-hf ran on a local Windows RTX 3060 with PyTorch 2.10.0+cu128,
Transformers 5.13.0 at a pinned source commit, and kaldialign 0.12.0.
The pinned official Open TTS transcriber and normalizer/scoring functions were
used without transcript correction or reference-text prompts.

Consecutive manifests contain up to 24 corpus rows; official duration sorting
runs **within each manifest**, with batch 8, BF16, at most 256 new tokens,
a 30-second audio cap and TF32 disabled. This is **not** the official
whole-manifest/global-duration-order default-batch-32 execution.

This is a **local reproduction**, not an official submission, leaderboard rank
or independent evaluation. The corpus has been used in development;
training-disjointness and unseen generalization are not established by this run.
WER does not establish human-perceived quality.

The historical six-voice result used a different host/PyTorch 2.11 stack.
The local source/loader environment also differs. Even with unchanged synthesis
weights, these are not controlled matched executions: do not attribute a small
WER difference to a model improvement or one specific software component.

## Identity And Related Documentation

The [machine-readable evidence](evidence/evaluation-update-20261112.json)
records exact counts, all seven official source hashes, ASR asset identities,
runtime versions and the public protocol digest.
Core weights SHA256: `8460e7d54a2fbf7fb54b3c7be7569c493c852de9a2aa1a0fd9fd9a099de581ed`.
Official source commit: `97109c086cf9a9c440e2b89a880aa1dcad567b4b`.
Public protocol SHA256: `6c9988b186523ed20533fe837f3bdde0082638d30b0793ef6322c5156e08e0f8`.

The already-published [single-thread CPU measurement](single-thread-cpu.md)
covers timing, frontend size and limitations; it is not new evidence in this note.
See also the [language-adaptation guide](language-adaptation/index.md) and
[full public model protocol](https://huggingface.co/owensong/Inflect-Mini-v3/blob/main/EVALUATION.md).
