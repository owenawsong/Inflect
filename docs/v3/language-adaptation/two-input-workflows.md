# Two Separate Aligned-Data Workflows

Choose the deployment input **before** recording and external alignment.
These are interfaces for a reviewed real-data pilot, not a passed Polish
experiment or qualified training preset. Follow the
[pinned-base steps](pinned-base.md) and
[coordinated vocabulary contract](coordinated-vocabulary.md) first.

For both branches, `./work/adaptation` must exist and every new output must
not exist. Keep original recordings and the base package unchanged. The real
source manifests must supply audio paths, exact transcripts, independent
groups, source/consent assertions and reviewed external timing labels. Counts
must be positive integers for every final codepoint and both boundary tokens,
with exactly600 waveform samples/frame at24k mono. Timed labels must satisfy
the [documented token/clock schema](data-and-validation.md#timed-alignment-instead-of-counts).
There is no bundled aligner/converter and no guessed or transplanted English
duration-label fallback.

The commands below are PowerShell examples. Supply approved `$PilotSteps`,
`$PilotLearningRate`, `$MaxFrames`, and selected checkpoint paths yourself.
No values for them are qualified by this guide. The explicit migration/map
files shown are conditional: omit `--vocab-migration` entirely if no additions
are required, and omit `--phone-mapping` consistently if no map is required.
`reviewed-vocab.json` is then the unchanged extracted base map.

## A. Reviewed External Phones: Phoneme-Input Export

`pl-reviewed-phones-aligned.jsonl` supplies caller `phones` and exact final
`alignment_phones`, alongside real labels and audio. Its source strings may
pass through the explicit reviewed map; labels refer to the resulting final
codepoints. Arbitrary G2P code is external to this CLI.

```powershell
python -B -m finetune_v3 coverage --manifest ./data/pl-reviewed-phones-aligned.jsonl --vocab ./work/adaptation/reviewed-vocab.json --base-vocab ./work/adaptation/base-vocab.json --phone-mapping ./work/adaptation/pl-phone-map.json --language pl --frontend prephonemized --output ./work/adaptation/pl-phones-coverage.json
python -B -m finetune_v3 prepare --manifest ./data/pl-reviewed-phones-aligned.jsonl --audio-root ./data/pl-audio --output ./work/adaptation/pl-phones-prepared --vocab ./work/adaptation/reviewed-vocab.json --base-vocab ./work/adaptation/base-vocab.json --phone-mapping ./work/adaptation/pl-phone-map.json --language pl --frontend prephonemized --split-seed 2026 --validation-fraction 0.1
python -B -m finetune_v3 audit --dataset ./work/adaptation/pl-phones-prepared/dataset.json
python -B -m finetune_v3 train --model ./inflect-mini-v3 --dataset ./work/adaptation/pl-phones-prepared/dataset.json --output ./work/adaptation/pl-phones-run --voice cassidy --device cpu --groups text_encoder duration pitch_noise speaker --vocab-migration ./work/adaptation/pl-migration.json --steps $PilotSteps --learning-rate $PilotLearningRate --max-frames $MaxFrames
python -B -m finetune_v3 export --run ./work/adaptation/pl-phones-run --checkpoint $PhoneCheckpoint --output ./work/adaptation/pl-phones-export
python -B -m finetune_v3 say --model ./work/adaptation/pl-phones-export --phonemes "$ReviewedInferencePhones" --output ./work/adaptation/pl-phones-check.wav
```

`$PhoneCheckpoint` must belong to `pl-phones-run/checkpoints` and retain its
matching controller sidecar. `$ReviewedInferencePhones` is the full final
mapped phone string. The CLI does not automatically apply the training phone
map to explicit `--phonemes`; supply the same final representation yourself.
This export intentionally refuses `--text`; it is not a raw-text Polish model.

## B. Exact eSpeak Phones: Raw-Text Export

First review eSpeak output for the target language and representation. Use
`pl-prompts.jsonl` for label-free coverage. Independently obtain real phone
alignment against the same final output. The separate
`pl-espeak-aligned.jsonl` must have `alignment_phones` but **no caller
phones/phonemes override**. Do not copy timing from coverage or a generated
English smoke. The language code alone does not certify correct pronunciation.

```powershell
python -B -m finetune_v3 coverage --manifest ./data/pl-prompts.jsonl --vocab ./work/adaptation/reviewed-vocab.json --base-vocab ./work/adaptation/base-vocab.json --phone-mapping ./work/adaptation/pl-phone-map.json --language pl --frontend espeak --output ./work/adaptation/pl-espeak-coverage.json
python -B -m finetune_v3 prepare --manifest ./data/pl-espeak-aligned.jsonl --audio-root ./data/pl-audio --output ./work/adaptation/pl-espeak-prepared --vocab ./work/adaptation/reviewed-vocab.json --base-vocab ./work/adaptation/base-vocab.json --phone-mapping ./work/adaptation/pl-phone-map.json --language pl --frontend espeak --split-seed 2026 --validation-fraction 0.1
python -B -m finetune_v3 audit --dataset ./work/adaptation/pl-espeak-prepared/dataset.json
python -B -m finetune_v3 train --model ./inflect-mini-v3 --dataset ./work/adaptation/pl-espeak-prepared/dataset.json --output ./work/adaptation/pl-espeak-run --voice cassidy --device cpu --groups text_encoder duration pitch_noise speaker --vocab-migration ./work/adaptation/pl-migration.json --steps $PilotSteps --learning-rate $PilotLearningRate --max-frames $MaxFrames
python -B -m finetune_v3 export --run ./work/adaptation/pl-espeak-run --checkpoint $EspeakCheckpoint --output ./work/adaptation/pl-espeak-export
python -B -m finetune_v3 say --model ./work/adaptation/pl-espeak-export --text "Proszę zaczekać." --output ./work/adaptation/pl-espeak-check.wav
```

`$EspeakCheckpoint` belongs to the separate `pl-espeak-run`. Export retains
the saved frontend/map/vocabulary and target Cassidy slot. Raw-text `say`
requires the saved engine version and installed matching language resources,
with no English fallback. Pin and qualify those external resources yourself;
the export does not bundle a complete language frontend environment.

Both branches default to the exported target voice and reject another slot
as unqualified. Exports require the complete source package's license/notices.
Neither branch includes events, VoiceTuner or enhancement. Select checkpoints
using held-out predicted-duration synthesis and fluent review, not only
alignment-conditioned reconstruction loss. See
[version-specific tested scope](implementation-status.md): software-model,
export and installed-wheel results are separate from Polish-quality
qualification. Neither those results nor these command/schema checks establish
that a Polish adaptation sounds correct.
