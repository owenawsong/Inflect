# Run Inflect v3 locally

[Back to Inflect](../../README.md) | [Features](features.md) | [Evaluation](evaluation.md)

## Before you start

- Use Python 3.12 in a fresh environment.
- Download the complete model package, not an individual weight file.
- Keep output WAVs, voice profiles and environments outside the downloaded package.
- Choose a new output filename. The launchers protect package contents and may reject overwrites.
- First-time setup needs network access. Preset speech runs locally after setup.

### Windows PowerShell

```powershell
py -3.12 -m venv inflect-env
.\inflect-env\Scripts\Activate.ps1
```

### Linux

```sh
python3.12 -m venv inflect-env
source inflect-env/bin/activate
```

## Mini: native 24 kHz

```sh
python -m pip install huggingface-hub==0.36.2
hf download owensong/Inflect-Mini-v3 --local-dir inflect-mini-v3
cd inflect-mini-v3
python verify_release.py
python -m pip install -r core/requirements-cpu.txt
python run.py speech --profile base --voice cassidy --text "A small idea can become something worth hearing." --subtype PCM_16 --output ../mini-cassidy.wav
```

Voice IDs: `cassidy`, `mark`, `hope`, `james`, `sophia`, `jasper`.
Use `python run.py speech --help` to inspect the installed package's controls.

## Mini: optimized CUDA

Use a separate fresh Python 3.12 environment with a compatible NVIDIA driver.
After downloading and entering Mini:

```sh
python -m pip install -r core/requirements-cuda.txt
python run.py speech --device cuda --runtime cuda-source --profile base --voice cassidy --text "A small idea can become something worth hearing." --subtype PCM_16 --output ../mini-cuda.wav
```

Do not combine the CPU and CUDA dependency recipes in one environment. The
optimized CUDA runtime and canonical CPU runtime are different execution paths.

## Micro

Start in another fresh Python 3.12 environment:

```sh
python -m pip install huggingface-hub==0.36.2
hf download owensong/Inflect-Micro-v3 --local-dir inflect-micro-v3
cd inflect-micro-v3
python -m pip install -r requirements-cpu.txt
python synthesize.py --verify
python synthesize.py --model micro --voice cassidy --text "A small idea can become something worth hearing." --output ../micro-cassidy.wav
```

Micro requires `--model micro`. Its package does not include Mini's event,
enhancement or reference-tuning components. CUDA needs a compatible separate
PyTorch/CUDA installation; the provided CPU recipe is not a CUDA recipe.

## Atom

Start in another fresh Python 3.12 environment:

```sh
python -m pip install huggingface-hub==0.36.2
hf download owensong/Inflect-Atom-v3 --local-dir inflect-atom-v3
cd inflect-atom-v3
python -m pip install -r requirements.txt
python -m pip install vendor/en_core_web_sm-3.8.0-py3-none-any.whl
python synthesize.py --verify
python synthesize.py --voice cassidy --text "A small idea can become something worth hearing." --output ../atom-cassidy.wav
```

Atom has **no `--model atom` flag**. Its own wrapper selects the model. The
published path uses a tiny learned core and signal-processing waveform stage.

## Troubleshooting

**Package verification fails:** use a new complete download directory. Do not
mix files from different revisions or remove files listed by its manifest.

**Python or dependency errors:** confirm Python 3.12 and use a fresh environment
for the selected model and CPU/CUDA recipe. Include versions in bug reports.

**Output path rejected:** choose a fresh `.wav` filename outside the package in
an existing directory. Never write outputs into a model directory.

**CUDA is unavailable:** verify the NVIDIA driver and CUDA-compatible PyTorch
installation. A CPU wheel cannot become CUDA-capable by changing a flag.

**A name or number sounds wrong:** inspect the spoken text and include the exact
input, voice, model revision and command when reporting the issue. Speech
recognition scores do not replace listening.

**A Mini feature flag fails in Micro or Atom:** optional Mini features do not
belong to those runtimes. Follow the command for the package you downloaded.

**The old GitHub example runs v2:** the original repository library and examples
are preserved for older integrations. Use the self-contained v3 model packages
in this guide rather than the legacy `inference.py` interface.
