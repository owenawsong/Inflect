#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ZIPVOICE_DIR = REPO_ROOT / "ZipVoice-official"
HF_ZIPVOICE_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--k2-fsa--ZipVoice" / "snapshots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare VoxCPM for ZipVoice fine-tuning and launch a 3060-safe teacher fine-tune."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "voxcpm_dataset" / "20260411_large_text_v1",
        help="Dataset version directory containing metadata.csv and audio/.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "inflect_finetune" / "teacher_foundation_v1",
        help="Working directory for manifests, fbank features, and experiments.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=REPO_ROOT / ".venv-voxcpm" / "Scripts" / "python.exe",
        help="Python executable to use for prep and training subprocesses.",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=0,
        help="Start stage: 0=split TSV, 1=prepare manifests, 2=prepare tokens, 3=compute fbank, 4=train, 5=average.",
    )
    parser.add_argument(
        "--stop-stage",
        type=int,
        default=4,
        help="Stop stage.",
    )
    parser.add_argument(
        "--preset",
        choices=["smoke", "demo", "v1"],
        default="v1",
        help="Preset for training defaults.",
    )
    parser.add_argument("--dev-ratio", type=float, default=0.02)
    parser.add_argument("--min-dev-per-voice", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-jobs", type=int, default=4)
    parser.add_argument(
        "--token-jobs",
        type=int,
        default=1,
        help="Token-prep jobs. Keep this at 1 on Windows to avoid Lhotse process-pool failures.",
    )
    parser.add_argument("--master-port", type=int, default=12379)
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--save-every-n", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--max-len", type=float, default=20.0)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument(
        "--use-fp16",
        type=int,
        choices=[0, 1],
        default=0,
        help="Mixed precision training. Default 0 because fp16 was unstable in local fine-tuning on this setup.",
    )
    parser.add_argument(
        "--condition-drop-ratio",
        type=float,
        default=0.1,
        help="Text condition dropout ratio during training. Lower is safer for conservative fine-tuning.",
    )
    parser.add_argument(
        "--skip-bad-batches",
        type=int,
        choices=[0, 1],
        default=0,
        help="Skip non-finite optimizer-step batches instead of aborting the run.",
    )
    parser.add_argument(
        "--clipping-scale",
        type=float,
        default=1.0,
        help="Gradient clipping scale for ScaledAdam. Lower than upstream default for safer fine-tuning.",
    )
    parser.add_argument(
        "--debug-nonfinite",
        type=int,
        choices=[0, 1],
        default=0,
        help="Enable detailed diagnostics for non-finite losses or gradients.",
    )
    parser.add_argument(
        "--return-cuts",
        type=int,
        choices=[0, 1],
        default=1,
        help="Return cut objects in the dataloader so bad batches can be traced back to exact examples.",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override path to base ZipVoice model.pt teacher checkpoint.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Override path to base ZipVoice model.json.",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=None,
        help="Override path to base ZipVoice tokens.txt.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="inflect_teacher_voxcpm_v1",
        help="Experiment subdirectory name under work-dir/exp.",
    )
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    if os.name == "nt":
        args.num_jobs = 1
        args.token_jobs = 1
    if args.preset == "smoke":
        args.num_iters = args.num_iters or 100
        args.save_every_n = args.save_every_n or 50
        args.log_interval = args.log_interval or 10
        args.max_duration = args.max_duration or 24
        args.num_workers = args.num_workers or 1
    elif args.preset == "demo":
        args.num_iters = args.num_iters or 1500
        args.save_every_n = args.save_every_n or 250
        args.log_interval = args.log_interval or 10
        args.max_duration = args.max_duration or 24
        args.num_workers = args.num_workers or 2
    else:
        args.num_iters = args.num_iters or 4000
        args.save_every_n = args.save_every_n or 500
        args.log_interval = args.log_interval or 25
        args.max_duration = args.max_duration or 32
        args.num_workers = args.num_workers or 2


def resolve_base_zipvoice_assets(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.checkpoint and args.model_config and args.token_file:
        return args.checkpoint.resolve(), args.model_config.resolve(), args.token_file.resolve()

    candidates = sorted(HF_ZIPVOICE_CACHE.glob("*/zipvoice/model.pt"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find cached base ZipVoice model.pt under Hugging Face cache. "
            "Run ZipVoice once or pass --checkpoint/--model-config/--token-file explicitly."
        )

    checkpoint = candidates[-1].resolve()
    base_dir = checkpoint.parent
    model_config = (base_dir / "model.json").resolve()
    token_file = (base_dir / "tokens.txt").resolve()
    if not model_config.is_file() or not token_file.is_file():
        raise FileNotFoundError(f"Incomplete base ZipVoice asset set under {base_dir}")
    return checkpoint, model_config, token_file


def ensure_python(python_path: Path) -> Path:
    python_path = python_path.resolve()
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable not found: {python_path}")
    return python_path


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    merged_env["PYTHONUTF8"] = "1"
    if env:
        merged_env.update(env)
    print("\n=== Running ===")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged_env, check=True)


def zipvoice_env() -> dict[str, str]:
    env = os.environ.copy()
    # Run official ZipVoice tools from their repo root so the PyPI `inflect`
    # package is imported instead of this repo's local `inflect/` package.
    env["PYTHONPATH"] = str(ZIPVOICE_DIR)
    return env


def main() -> None:
    args = parse_args()
    apply_preset(args)

    python_exe = ensure_python(args.python)
    dataset_dir = args.dataset_dir.resolve()
    work_dir = args.work_dir.resolve()
    raw_dir = work_dir / "data" / "raw"
    manifests_dir = work_dir / "data" / "manifests"
    fbank_dir = work_dir / "data" / "fbank"
    exp_dir = work_dir / "exp" / args.exp_name

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    fbank_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, model_config, token_file = resolve_base_zipvoice_assets(args)
    env = zipvoice_env()

    train_tsv = raw_dir / "custom_train.tsv"
    dev_tsv = raw_dir / "custom_dev.tsv"
    raw_train_manifest = manifests_dir / "custom-finetune_cuts_raw_train.jsonl.gz"
    raw_dev_manifest = manifests_dir / "custom-finetune_cuts_raw_dev.jsonl.gz"
    train_manifest = manifests_dir / "custom-finetune_cuts_train.jsonl.gz"
    dev_manifest = manifests_dir / "custom-finetune_cuts_dev.jsonl.gz"
    train_fbank = fbank_dir / "custom-finetune_cuts_train.jsonl.gz"
    dev_fbank = fbank_dir / "custom-finetune_cuts_dev.jsonl.gz"

    if args.start_stage <= 0 <= args.stop_stage:
        run(
            [
                str(python_exe),
                str(REPO_ROOT / "scripts" / "prepare_inflect_finetune_data.py"),
                "--dataset-dir",
                str(dataset_dir),
                "--out-dir",
                str(raw_dir),
                "--dev-ratio",
                str(args.dev_ratio),
                "--min-dev-per-voice",
                str(args.min_dev_per_voice),
                "--seed",
                str(args.seed),
            ]
        )

    if args.start_stage <= 1 <= args.stop_stage:
        for subset, tsv_path in (("raw_train", train_tsv), ("raw_dev", dev_tsv)):
            run(
                [
                    str(python_exe),
                    "-m",
                    "zipvoice.bin.prepare_dataset",
                    "--tsv-path",
                    str(tsv_path),
                    "--prefix",
                    "custom-finetune",
                    "--subset",
                    subset,
                    "--num-jobs",
                    str(args.num_jobs),
                    "--output-dir",
                    str(manifests_dir),
                ],
                cwd=ZIPVOICE_DIR,
                env=env,
            )

    if args.start_stage <= 2 <= args.stop_stage:
        for input_path, output_path in (
            (raw_train_manifest, train_manifest),
            (raw_dev_manifest, dev_manifest),
        ):
            run(
                [
                    str(python_exe),
                    str(REPO_ROOT / "scripts" / "prepare_tokens_serial.py"),
                    "--input-file",
                    str(input_path),
                    "--output-file",
                    str(output_path),
                    "--tokenizer",
                    "emilia",
                    "--lang",
                    "default",
                ],
            )

    if args.start_stage <= 3 <= args.stop_stage:
        for subset in ("train", "dev"):
            run(
                [
                    str(python_exe),
                    "-m",
                    "zipvoice.bin.compute_fbank",
                    "--source-dir",
                    str(manifests_dir),
                    "--dest-dir",
                    str(fbank_dir),
                    "--dataset",
                    "custom-finetune",
                    "--subset",
                    subset,
                    "--num-jobs",
                    str(args.num_jobs),
                ],
                cwd=ZIPVOICE_DIR,
                env=env,
            )

    if args.start_stage <= 4 <= args.stop_stage:
        run(
            [
                str(python_exe),
                "-m",
                "zipvoice.bin.train_zipvoice",
                "--world-size",
                "1",
                "--master-port",
                str(args.master_port),
                "--use-fp16",
                str(args.use_fp16),
                "--finetune",
                "1",
                "--skip-bad-batches",
                str(args.skip_bad_batches),
                "--clipping-scale",
                str(args.clipping_scale),
                "--debug-nonfinite",
                str(args.debug_nonfinite),
                "--base-lr",
                str(args.base_lr),
                "--condition-drop-ratio",
                str(args.condition_drop_ratio),
                "--num-iters",
                str(args.num_iters),
                "--save-every-n",
                str(args.save_every_n),
                "--log-interval",
                str(args.log_interval),
                "--max-duration",
                str(args.max_duration),
                "--max-len",
                str(args.max_len),
                "--num-workers",
                str(args.num_workers),
                "--return-cuts",
                str(args.return_cuts),
                "--model-config",
                str(model_config),
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                "emilia",
                "--lang",
                "default",
                "--token-file",
                str(token_file),
                "--dataset",
                "custom",
                "--train-manifest",
                str(train_fbank),
                "--dev-manifest",
                str(dev_fbank),
                "--exp-dir",
                str(exp_dir),
            ],
            cwd=ZIPVOICE_DIR,
            env=env,
        )

    if args.start_stage <= 5 <= args.stop_stage:
        run(
            [
                str(python_exe),
                "-m",
                "zipvoice.bin.generate_averaged_model",
                "--iter",
                str(args.num_iters),
                "--avg",
                "2",
                "--model-name",
                "zipvoice",
                "--exp-dir",
                str(exp_dir),
            ],
            cwd=ZIPVOICE_DIR,
            env=env,
        )

    print("\nDone.")
    print(f"Working directory: {work_dir}")
    print(f"Experiment dir:    {exp_dir}")
    print(f"Base checkpoint:   {checkpoint}")
    print(f"Model config:      {model_config}")
    print(f"Token file:        {token_file}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nSubprocess failed with exit code {exc.returncode}.")
        sys.exit(exc.returncode)
