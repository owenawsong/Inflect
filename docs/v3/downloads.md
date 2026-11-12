# Download Inflect v3

[Project home](../../README.md) | [Full installation guide](getting-started.md) | [Playground](https://huggingface.co/spaces/owensong/Inflect-v3)

**The runnable v3 code is bundled with each model on Hugging Face.**
You do not need to clone GitHub to generate v3 speech. The GitHub `inflect/`
library and older examples are v1/v2 interfaces, not the v3 entry point.
Download the complete model repository, not an individual checkpoint.

| Start with | Complete package | Files you should see |
| :--- | :--- | :--- |
| **Mini: quality and features** | [Mini download](https://huggingface.co/owensong/Inflect-Mini-v3/tree/main) | `run.py`, `verify_release.py`, `core/`, `extras/` |
| **Micro: smaller neural model** | [Micro download](https://huggingface.co/owensong/Inflect-Micro-v3/tree/main) | `synthesize.py`, `requirements-cpu.txt`, `release-manifest.json` |
| **Atom: smallest learned core** | [Atom download](https://huggingface.co/owensong/Inflect-Atom-v3/tree/main) | `synthesize.py`, `requirements.txt`, `manifest.json` |

## Recommended: Mini

Use a fresh **Python 3.12** environment outside the downloaded package.
Start from the parent directory where you want the model folder:

```sh
python -m pip install huggingface-hub==0.36.2
hf download owensong/Inflect-Mini-v3 --local-dir inflect-mini-v3
cd inflect-mini-v3
python verify_release.py
python -m pip install -r core/requirements-cpu.txt
python run.py speech --profile base --voice cassidy --text "A small idea can become something worth hearing." --subtype PCM_16 --output ../my-first-mini.wav
```

The output is native 24 kHz audio. Use a new output filename outside the package.
Use the [installation guide](getting-started.md) for Windows environment setup,
Linux setup, Mini CUDA, and the separate Micro/Atom generation commands.

## Download a smaller model

Run either command from your chosen parent directory, not inside another model:

```sh
hf download owensong/Inflect-Micro-v3 --local-dir inflect-micro-v3
hf download owensong/Inflect-Atom-v3 --local-dir inflect-atom-v3
```

Their dependency files and commands differ from Mini. Follow
[Micro setup](getting-started.md#micro) or [Atom setup](getting-started.md#atom).
Mini's event tags, enhancement and Voice Tuner are not included in the smaller models.

## Update an existing download

Repeat your model's `hf download` command from the directory containing its
folder, then run its verification command. Update the entire package so the
runtime, assets and manifests stay together. Keep your environment, generated
audio and voice profiles outside that folder. Do not mix v2 and v3 files.

For Mini's eight event tags, [install the event dependencies and use the event command](features.md#vocal-event-tags).
