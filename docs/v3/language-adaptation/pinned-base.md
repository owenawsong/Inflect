# Obtain and Verify the Complete Mini Base

Use Python3.12 with your separately chosen CPU/CUDA PyTorch environment, then
install the toolkit from the repository root:

```sh
python -m pip install './finetune_v3'
```

The toolkit does not include model weights. The following is the complete
official [Inflect Mini v3 snapshot](https://huggingface.co/owensong/Inflect-Mini-v3/tree/0490aee6a9c66f9375b48287a2c5e5788afd00cc), pinned to
`0490aee6a9c66f9375b48287a2c5e5788afd00cc` rather than a moving branch:

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='owensong/Inflect-Mini-v3', revision='0490aee6a9c66f9375b48287a2c5e5788afd00cc', local_dir='inflect-mini-v3')"
python -B ./inflect-mini-v3/verify_release.py --root ./inflect-mini-v3
```

The first command downloads files and requires network access and space for
the full snapshot. It is an instruction for a new installation, not a step
already performed by this documentation audit. Use a fresh destination.
Do not apply file filters or download only `weights.pt`: retain the root
verification script, inventories, code, license documents and core notices.
The second command is the actual packaged verifier's `--root` interface. It
uses the standard library to check root-inventory file sizes and hashes; it
does not load a model or download dependencies. Stop if it fails.

Keep that pinned directory unchanged. Create an independent work directory,
then extract the actual base vocabulary:

```sh
mkdir -p ./work/adaptation
python -B -m finetune_v3 vocab --model ./inflect-mini-v3 --output ./work/adaptation/base-vocab.json
```

`mkdir -p` is POSIX shell syntax. In PowerShell, instead use
`New-Item -ItemType Directory -Force ./work/adaptation`; the Python commands
are otherwise the same. All generated output paths must be new and outside
the verified base. `vocab` loads the base payload to extract its mapping;
root verification and label-free coverage are different operations.

The tested base-core weight SHA256 is
`8460e7d54a2fbf7fb54b3c7be7569c493c852de9a2aa1a0fd9fd9a099de581ed`.
Root inventory validity, toolkit installation, and vocabulary extraction are
distinct software checks, not language-quality evidence. Consult the
version-specific [tested scope](implementation-status.md) for actual
model, export and installed-wheel results.
