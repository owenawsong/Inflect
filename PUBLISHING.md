# Publishing

This repo mixes publishable code with large local artifacts. The intended split is:

- GitHub:
  - source code
  - small docs
  - scripts
  - patch files for local dependencies
- Hugging Face Dataset Hub:
  - VoxCPM dataset versions in AudioFolder format
- Local only:
  - `outputs/`
  - `reference_voices/`
  - checkpoints
  - virtual environments
  - vendored experimental repos

## GitHub

The root `.gitignore` intentionally keeps local artifacts out of Git.

The current local ZipVoice fixes are preserved in:

- [patches/zipvoice-local-fixes.patch](C:/users/owen/Inflect-New/patches/zipvoice-local-fixes.patch)

Those fixes should be applied to a local `ZipVoice-official` checkout before running the teacher fine-tune workflow.

## Hugging Face Dataset Upload

Use:

```powershell
cd C:\Users\Owen\Inflect-New
$env:HF_TOKEN = "your_token_here"
.\.venv-voxcpm\Scripts\python.exe scripts\upload_voxcpm_dataset.py --dataset-dir outputs\voxcpm_dataset\20260411_large_text_v1 --repo-id your-name/voxcpm2-synthetic-v1 --private
```

The upload script expects the dataset directory to contain:

- `metadata.csv`
- `README.md`
- `audio/`

The current VoxCPM dataset format is already compatible with Hugging Face `AudioFolder`.

## Recommended Publish Order

1. Commit publishable repo changes to GitHub.
2. Upload the current VoxCPM dataset privately to Hugging Face.
3. Keep growing the local dataset or create a clean `v2` dataset later if needed.

## Notes

- Do not commit `outputs/` to the repo.
- Do not commit reference voice audio to the repo.
- Do not commit full nested external repos unless you intentionally want to vendor them.
