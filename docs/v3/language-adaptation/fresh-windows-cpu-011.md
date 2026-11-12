# Fresh Windows CPU Environment Qualification

On 2026-11-12, toolkit **0.1.1** completed an installed CLI smoke in a new
Windows x64 virtual environment, without system site-packages. This was a fresh
dependency installation on an existing host, not a fresh Windows installation.
The same final wheel was tested; these results do not transfer to changed bytes
or to the separately staged 0.1.2 implementation.

| Component | Observed version |
| --- | --- |
| Python | CPython 3.12.11 |
| PyTorch | 2.6.0+cpu; CUDA version absent and CUDA remained uninitialized |
| Transformers / tokenizers | 5.2.0 / 0.22.2 |
| NumPy / SciPy / SoundFile | 1.26.4 / 1.17.1 / 0.13.1 |
| phonemizer-fork / espeakng-loader | 3.3.2 / 0.2.4 |
| Actual eSpeak NG engine | 1.52.0; newly installed DLL and seven English/shared assets hash-checked |
| pip / pytest installed | 25.0.1 / 9.1.1 |

Declared dependencies were resolved in one successful pip installation using
PyPI and the official PyTorch CPU index, constraints and `--no-cache-dir`.
`pip check` reported no broken requirements. The original pip report retained
selected artifact versions, URLs and SHA256 values. Qualification checked these
against installed distributions, module locations and the toolkit wheel/RECORD;
imports came from the new environment, not a checkout or another site-packages.
This verifies reported artifact provenance, not every HTTP redirect or an OS
network firewall. Post-install checks and functional execution denied Python
network attempts and did not download models.

### What Actually Passed

From an unrelated working directory, the installed CLI prepared and audited
six existing self-generated English mechanical clips, completed **one CPU
optimizer update**, exported its controller-validated checkpoint with notices,
and generated raw-text audio through the restored eSpeak frontend. Final checks
verified native24 mono PCM16, **120 frames / 72,000 samples / 3 seconds**, finite
audio below full scale, and unchanged source base weights. The functional run
completed 15:25:10-15:29:04 UTC in 234.219 seconds.

The full pytest suite was **not** run in this fresh environment. The separate
earlier 199-passed/1-skipped source-suite result used existing dependencies.
The recovery wrappers separately passed synthetic policy/admission tests;
those are not additional fresh-environment model tests.

### Exercised Commands

Paths below are normalized names for the exact exercised inputs and new outputs,
not a claim that these input files ship with the wheel. Supply the reviewed
base snapshot, vocab and six mechanical fixtures with their real aligned clocks;
do not invent alignment for a recording. `PY` denotes the fresh environment's
Python executable. The functional commands below are equivalent CLI forms;
the actual run dispatched the installed CLI through the bound offline checker
under `-I -X utf8 -B`. Use a new output location rather than overwriting a run.
The original successful dependency installation used the following normalized
command. Its extra CPU constraints were `torch==2.6.0+cpu` and `pytest==9.1.1`,
alongside the pinned requirements-tested snapshot. This is an observed resolver
result, not a complete cross-platform dependency lock.

```text
PY -I -B -m pip --isolated --disable-pip-version-check --no-input --keyring-provider disabled install --no-cache-dir --no-compile --retries 0 --timeout 60 --only-binary=:all: --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --constraint ./requirements-tested.txt --constraint ./cpu-constraints.txt --report ./install-report.json ./inflect_mini_v3_adapt-0.1.1-py3-none-any.whl pytest==9.1.1
PY -I -B -m pip --isolated check
PY -I -X utf8 -B -m pip --isolated --disable-pip-version-check inspect --local
PY -I -X utf8 -B -m finetune_v3 prepare --manifest ./mechanical-input/metadata.jsonl --audio-root ./mechanical-input --output ./work/prepared --vocab ./mechanical-input/vocab.json --language en-us --frontend espeak --split-seed 2026 --validation-fraction 0.1
PY -I -X utf8 -B -m finetune_v3 audit --dataset ./work/prepared/dataset.json
PY -I -X utf8 -B -m finetune_v3 train --model ./inflect-mini-v3 --dataset ./work/prepared/dataset.json --output ./work/adapt-run --voice cassidy --device cpu --groups text_encoder duration pitch_noise speaker --steps 1 --learning-rate 0.00001 --gradient-accumulation 1 --checkpoint-every 1 --validation-every 1 --max-validation-clips 1 --max-frames 120 --seed 2026 --threads 2
PY -I -X utf8 -B -m finetune_v3 export --run ./work/adapt-run --checkpoint ./work/adapt-run/checkpoints/step-0000001.pt --output ./work/export
PY -I -X utf8 -B -m finetune_v3 say --model ./work/export --text "A small model can still speak clearly." --output ./work/rawtext.wav --seed 42
```

These tiny counts are test settings, not recommended adaptation hyperparameters.
This Windows-only English mechanical check establishes neither Polish training,
new-language pronunciation, listening quality, Linux support nor CUDA training.

### Preserved Failures and Recovery

The first setup qualification failed after installation succeeded: `pip inspect`
could not print U+2192 under Windows cp1252. Explicit `-X utf8` corrected inventory
output; `PYTHONUTF8` alone would be ignored by `-I`. A subsequent provenance
check rejected the selected Torch URL's `download-r2.pytorch.org` host because
the original policy allowed only `download.pytorch.org`. The official CPU index
was then inspected and independently matched the **complete selected URL,
version and hash**. A separately admitted exact-tuple exception recovered
qualification without a general host exception, reinstall or dependency change.
Both original failed receipts remain failed; r004 recovery and r005 functional
success have distinct receipts. No deterministic-training policy was relaxed.

Artifact pins retained with this qualification:

- Toolkit wheel: `07c917b0dee6eb4961947ff032c0bfd5b709e7adf9b49e4e7614ed1d949272d5`.
- Original pip report: `babd4b6322e24b37a7cd44f748ff22ada8247daf551e0daf6209a941b1c16d91`.
- Torch CPU wheel: `4027d982eb2781c93825ab9527f17fbbb12dbabf422298e4b954be60016f87d8`.
- eSpeak DLL: `646d387acbc7ac2aa45e3625aa00a6835ae5d446ff8b0748298c3900b4dde258`.
- Recovery receipt: `80cbfa0beecdb0dfd5fe9cf809964e96606ea6e4234eb1133aad282701f4eea0`.
- Functional receipt: `d21c16d38b8b741c01f242ec04101a41af420b5cd14482e38ae36cecc7ab460c`.
