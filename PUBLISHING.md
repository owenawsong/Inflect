# Publishing

This project mixes source code, released model packages, large generated artifacts, and private research inputs. Public publishing has to separate those cleanly.

## Publish Split

### GitHub Should Contain

- source code
- scripts
- docs
- patch files for local external dependencies
- lightweight config and metadata files
- links to released Hugging Face models and datasets

### Hugging Face Models Should Contain

- runnable model weights
- model card
- inference entry points
- examples or sample gallery assets
- model-specific license and limitations

Current released model:

- [owensong/Inflect-Nano-v1](https://huggingface.co/owensong/Inflect-Nano-v1)

### Hugging Face Datasets Should Contain

- frozen synthetic or curated dataset snapshots
- dataset cards
- dataset-specific licensing and provenance notes
- generation settings when audio is teacher-generated

### Keep Local Only

- `outputs/`
- `reference_voices/`
- raw checkpoints
- virtual environments
- local external repos such as `ZipVoice-official/` and `third_party/`
- unfinished teacher-generation runs
- private reference audio

## Current Public Status

Inflect-Nano-v1 is released on Hugging Face as a complete 4.63M-parameter English text-to-waveform stack.

Inflect-Nano-v2 is active research. Do not publish v2 claims as release facts until a frozen checkpoint, sample gallery, evaluation notes, and model card are ready.

## Release Rules

- Do not commit model weights directly to GitHub.
- Do not commit generated audio datasets directly to GitHub.
- Do not label synthetic datasets `apache-2.0` unless the dataset license has been separately reviewed.
- Do not claim voice cloning for v1. The released v1 model is a single English male voice.
- Do not claim production quality. v1 is an experimental tiny-model release.
- Do not claim v2 is better than v1 until fixed-prompt listening and objective checks support it.

## Dataset Licensing

Dataset licensing is separate from repository licensing.

Reason:

- the repo code license is not automatically the dataset license
- synthetic audio can still inherit constraints from reference voices, prompts, or teacher models
- source voice rights and upstream model terms must be disclosed honestly

Safe default for public synthetic dataset cards:

- dataset card tag: `license: other`
- explicit release note explaining:
  - reference voice prompts are not bundled
  - generated audio is synthetic
  - source voice rights and upstream model terms still matter

## Release Checklist

Before publishing a model package to Hugging Face:

1. Freeze the exact checkpoint files.
2. Render a fixed sample gallery.
3. Run the objective diagnostics used by the project.
4. Write a model card with real limitations.
5. Verify clean install and CLI inference from a fresh environment.
6. Link the Hugging Face release from the GitHub README.

Before publishing a dataset package to Hugging Face:

1. Freeze a snapshot.
2. Remove private reference material.
3. Write a public dataset card.
4. Include provenance and generation settings.
5. Verify the manifest/audio layout loads cleanly.
6. Link the dataset only after the upload is complete.

## GitHub Cleanup Rules

- do not commit `outputs/`
- do not commit `reference_voices/`
- do not commit checkpoints
- do not commit `.pyc` or `__pycache__`
- do not commit full vendor repos unless intentionally vendoring them
- do not commit credentials, API keys, tokens, or private SSH material

## Recommended Publish Order

1. Update GitHub source/docs.
2. Freeze model or dataset artifact locally.
3. Build a clean Hugging Face package.
4. Test from a fresh clone or fresh environment.
5. Upload to Hugging Face.
6. Link the release from GitHub.
7. Only then announce or benchmark publicly.
