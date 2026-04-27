# Publishing

This project mixes real source code with large local artifacts and private research inputs. Public publishing has to separate those cleanly.

## Publish Split

### GitHub should contain

- source code
- scripts
- docs
- patch files for local external dependencies
- lightweight config and metadata files

### Hugging Face Datasets should contain

- prepared VoxCPM synthetic datasets in `AudioFolder` layout
- dataset cards
- dataset-specific licensing and provenance notes

### Keep local only

- `outputs/`
- `reference_voices/`
- checkpoints
- virtual environments
- local external repos such as `ZipVoice-official/` and `third_party/`

## Current GitHub State

The publish branch currently contains the release scaffold and should be expanded with the real source/docs set.

Branch:

- `codex/publish-scaffold`

Draft PR:

- [PR #1](https://github.com/owenawsong/Inflect/pull/1)

## Local ZipVoice Fixes

The current teacher fine-tuning path depends on local changes to ZipVoice that are not committed inside this repo's main tree because `ZipVoice-official/` is a nested local checkout.

Those changes are preserved here:

- [zipvoice-local-fixes.patch](patches/zipvoice-local-fixes.patch)

Apply that patch to a local `ZipVoice-official/` checkout before running the current fine-tuning workflow.

## Dataset Release Strategy

The active dataset run is:

- `outputs/voxcpm_dataset/20260411_large_text_v1`

It is usable, but it is an active working dataset rather than a clean frozen public snapshot.

The better public-release flow is:

1. freeze a snapshot
2. write a public dataset card
3. attach a clear license/provenance notice
4. upload that frozen snapshot to Hugging Face

Use:

- [prepare_public_voxcpm_dataset.py](scripts/prepare_public_voxcpm_dataset.py)
- [upload_voxcpm_dataset.py](scripts/upload_voxcpm_dataset.py)

## Dataset Licensing

Do not label the public dataset `apache-2.0`.

Reason:

- the repo code license is not automatically the dataset license
- the dataset is synthetic audio generated from reference voice prompts
- voice/source provenance is mixed and must be disclosed honestly

For the current public dataset release path, the safe default is:

- dataset card tag: `license: other`
- explicit release note explaining:
  - reference voice prompts are not bundled
  - generated audio is synthetic
  - source voice rights and upstream model terms still matter

## Uploading the Dataset to Hugging Face

Target username:

- `owensong`

Example public upload after preparing a snapshot:

```powershell
cd Inflect
$env:HF_TOKEN = "your_token_here"
.\.venv-voxcpm\Scripts\python.exe scripts\prepare_public_voxcpm_dataset.py --dataset-dir outputs\voxcpm_dataset\20260411_large_text_v1 --out-dir outputs\publish\voxcpm2_synthetic_en_v1_public
.\.venv-voxcpm\Scripts\python.exe scripts\upload_voxcpm_dataset.py --dataset-dir outputs\publish\voxcpm2_synthetic_en_v1_public --repo-id owensong/voxcpm2-synthetic-en-v1
```

## GitHub Cleanup Rules

- do not commit `outputs/`
- do not commit `reference_voices/`
- do not commit checkpoints
- do not commit `.pyc` or `__pycache__`
- do not commit full vendor repos unless intentionally vendoring them

## Recommended Publish Order

1. finish repo docs and structure cleanup
2. stage only publishable code/docs
3. push GitHub branch updates
4. freeze a dataset snapshot
5. upload the public dataset to Hugging Face
6. link the dataset from the GitHub README
