# Language Adaptation

Use the mechanically tested Mini v3 tools to build a separate fixed-target
adaptation from your own licensed speech and external phone alignment. A new
language's output quality is a pilot question, not a consequence of a working
training loop or an available G2P backend. Polish and multilingual quality have
not been qualified.

## Start Here

1. **Install and pin the base.** Follow the
   [toolkit front door](../../../finetune_v3/README.md#install-and-obtain-the-base).
   Use Python3.12 and a complete official Mini package, including its licenses.
2. **Check text before aligning audio.** Extract `vocab`, choose the language
   frontend, and run label-free `coverage` with the [command recipes](commands.md).
   Review unknown sounds and number/name readings before collecting more data.
3. **Prepare real supervision.** Follow [data and validation](data-and-validation.md)
   and the [exact row/clock contract](workflow.md#prepared-data-contract). Supply
   reviewed transcripts, external alignment, provenance, and independent groups.
4. **Run a representative pilot.** Use the [workflow](workflow.md) and
   [pilot/memory guide](pilot-and-training.md). Mechanical English smoke checks
   mechanics; held-out synthesis and fluent listening judge language quality.
5. **Export and verify.** Retain checkpoint sidecars, frontend identity, target
   voice, licenses, and final inventories. Use [tested scope and release gates](implementation-status.md)
   to distinguish demonstrated behavior from remaining work.

## Choose Your Input Path

| Path | Use it for | What you must provide |
| --- | --- | --- |
| eSpeak frontend | Raw text with an explicit supported language | Reviewed normalization/pronunciation, matching engine version and resources |
| Prephonemized | Custom G2P or manually reviewed phone strings | Full final phones at training and inference, plus external timing for recordings |
| Mechanical smoke | Testing prepare/train/resume/export/inference | A local base; generated English clocks are valid only for their own generated waveforms |

See [frontends and symbols](frontends-and-symbols.md) for custom phones and
unused-ID migration. The tools do not automatically align arbitrary recordings,
turn English checkpoints multilingual by changing a code, or add new voice slots.
V2 `inflect-adapt` commands and VITS presets are not Mini v3 training recipes.

## Tested Installation Scope

Version **0.1.1** passed the full CPU source suite: **199 passed, 1 skipped**,
including 42 real-model cases. Its new wheel was installed offline into an
isolated `--target` directory using existing qualified dependencies. From an
unrelated Temp working directory, the installed CLI exported the existing
English mechanical checkpoint and generated mono24 PCM16 raw-text output,
120 native frames. This was not a fresh complete-dependency installation or a
new-language quality test. See [exact tested scope and artifact pins](implementation-status.md).

The earlier **0.1.0/r004** installed-wheel result, including its 193-frame
mechanical output and original wheel hash, remains separate historical evidence.
Neither frame count is a language-quality comparison.

The tested public-core `weights.pt` SHA256 is:

```text
8460e7d54a2fbf7fb54b3c7be7569c493c852de9a2aa1a0fd9fd9a099de581ed
```

Keep the unchanged base and give each adaptation its own data version, run,
and export. Record the actual model and dataset checksums rather than only a
moving repository name.

## Further Reading

- [Polish considerations](polish.md): concrete contrasts and text-review tasks,
  not a qualified Polish recipe.
- [Pilot and training](pilot-and-training.md): memory measurement, checkpoint
  selection, full-corpus adaptation, and evaluation.
- [Implementation status](implementation-status.md): current tested boundaries
  and the gates for publishing an adapted model.
- [References](references.md): primary technical and linguistic sources.

Data quantity, learning rate, update count, and hardware depend on your corpus.
No universal successful preset is established here. Optional English events,
tuning, enhancement, and retained voices need their own post-adaptation checks.
