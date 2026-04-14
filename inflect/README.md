# Inflect Package Area

This folder contains Inflect-native code that sits outside the upstream ZipVoice checkout.

## `enhancer/`

Experimental enhancement stack intended to run after the backbone is finalized.

Current status:

- code exists
- full training is intentionally deferred
- should only be trained against the final chosen backbone outputs

Key files:

- `model.py`
- `dataset.py`
- `losses.py`
- `train.py`
- `infer.py`
- `configs/base.py`

## `para_module/`

Experimental paralinguistic sound generation branch.

Current status:

- still research-only
- not part of the current mainline roadmap
- kept for future extension work, not current backbone work

## `data/`

Helpers and assets for Inflect-side dataset handling and tag definitions.

Notes:

- `.pt` dataset artifacts are local and ignored
- extraction helpers are publishable source

## Main rule

Do not confuse `inflect/` with the main ZipVoice teacher workflow.

The current mainline is:

- upstream ZipVoice training/inference
- local runtime improvements
- VoxCPM-based teacher fine-tuning

`inflect/` is where the project-specific extension work lives.
