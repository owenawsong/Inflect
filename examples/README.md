# Examples

## Mini v3: Load Once, Render More Than One Utterance

Use a fresh Python 3.12 environment and a complete Hugging Face Mini package,
not this repository's legacy `inflect/` library. Follow the
[v3 installation guide](../docs/v3/getting-started.md) for dependencies.

```bash
python -m pip install huggingface-hub==0.36.2
hf download owensong/Inflect-Mini-v3 --revision 0490aee6a9c66f9375b48287a2c5e5788afd00cc --local-dir inflect-mini-v3
python -m pip install -r inflect-mini-v3/core/requirements-cpu.txt
mkdir generated
python -B examples/v3_reuse_mini.py --package inflect-mini-v3 --voice cassidy --text "A small idea can become something worth hearing." --text "The next sentence reuses the same loaded model." --output-dir generated
```

This writes new `speech-001.wav` and `speech-002.wav` files. The output directory
must already exist outside the entire package; existing output files are never
overwritten. Run from the GitHub checkout, not from inside the downloaded package.
The example never downloads model assets itself or guesses a cache snapshot path.

The example creates one `MiniSession` for the process. Its `render(text, voice)`
method returns complete WAV bytes using the same loaded model and numeric
normalizer for every call. Requests are serialized by a lock. Process exit owns
cleanup; loading another bundle/session in that process is deliberately refused.

Only **CPU, canonical runtime, base speech, native 24 kHz mono PCM16 and seed42**
are supported here. No Q1500/delivery profile, enhancement, tuning, event
generation, server or streaming interface is included. Text containing `[]<>`
is rejected, with a link to the
[official event CLI](../docs/v3/features.md#vocal-event-tags), not spoken literally.
This is an example using the bundled public runtime, not a general stable SDK.

Before importing any bundled code, it verifies the fixed core assembly SHA256
and every listed core file, including the synthesis weights. It does not pin
the mutable top-level card/README inventory. Use an unchanged source-only
package: symlinks/junctions, bundled code caches/extensions, unlisted Python
sources and preloaded colliding modules are refused. Missing dependencies fail
without installation or an automatic retry. Offline flags are not an OS-level
network isolation guarantee.

This reuses loaded state rather than respawning a model for every utterance;
it makes no measured speedup or cold-start/streaming latency claim.
CPU qualification on Windows 11 / AMD Ryzen 9 3900X used Python 3.12.11 and
PyTorch 2.11.0+cpu: two actual render calls reused one session, with each model,
runtime/export and numeric-normalizer loader observed once. Both WAVs were
byte-identical to independently spawned public CLI controls. Observational
profiling counted calls; it is not speedup evidence. See the small
[machine-readable qualification](../docs/v3/evidence/reuse-mini-20261112.json).
CUDA remains outside this example's supported scope.

## Legacy v2 Examples

The original Micro/Nano examples below are retained for v2, not v3 weights.

These examples use the exact self-contained packages published on Hugging
Face. Model weights are downloaded into the Hugging Face cache rather than
committed to GitHub.

### One model

```bash
python -m pip install -r requirements.txt
python examples/download_and_speak.py \
  --model micro \
  --text "A complete local voice can fit almost anywhere." \
  --output inflect.wav
```

Options:

```text
--model micro|nano
--device cpu|cuda
--speed 1.0
--variation 0.667
--seed 7
```

The model repository is cached after the first download. Long text is handled
by the punctuation-aware runtime included with the selected package.

### Direct comparison

```bash
python examples/compare_models.py \
  --text "Render both models with exactly the same controls." \
  --output-dir comparison
```

This writes:

```text
comparison/inflect-micro-v2.wav
comparison/inflect-nano-v2.wav
```

The comparison runs each self-contained model in its own Python process. This
avoids module-name collisions between two model packages while preserving the
same text, seed, speed, and variation.
