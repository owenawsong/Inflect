r"""
Run curated Phase 2 ZipVoice experiment sweeps.

Examples:
  .\.venv-voxcpm\Scripts\python.exe scripts\run_phase2_sweeps.py --suite tshift
  .\.venv-voxcpm\Scripts\python.exe scripts\run_phase2_sweeps.py --suite promptcap
  .\.venv-voxcpm\Scripts\python.exe scripts\run_phase2_sweeps.py --suite tracka --voices jessica --prompt-ids 06_emotional
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = PROJECT_ROOT / "scripts" / "test_base_zipvoice.py"
PYTHON = sys.executable


def run_one(args):
    cmd = [PYTHON, str(BENCH_SCRIPT), *args]
    print("\n=== Running ===")
    print(" ".join(cmd))
    print()
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["tshift", "promptcap", "phase2a", "tracka"])
    ap.add_argument("--voices", default=None, help="Optional comma-separated voice subset.")
    ap.add_argument("--prompt-ids", default=None, help="Optional comma-separated prompt subset.")
    ns = ap.parse_args()

    if ns.suite == "tshift":
        experiments = [
            ["--model-name", "zipvoice_distill", "--num-steps", "4", "--t-shift", "0.5"],
            ["--model-name", "zipvoice_distill", "--num-steps", "4", "--t-shift", "0.7"],
            ["--model-name", "zipvoice_distill", "--num-steps", "4", "--t-shift", "0.9"],
            ["--model-name", "zipvoice_distill", "--num-steps", "4", "--t-shift", "1.0"],
        ]
    elif ns.suite == "promptcap":
        experiments = [
            ["--model-name", "zipvoice_distill", "--num-steps", "4", "--prompt-duration-cap", "3"],
            ["--model-name", "zipvoice_distill", "--num-steps", "4", "--prompt-duration-cap", "5"],
            ["--model-name", "zipvoice_distill", "--num-steps", "4", "--prompt-duration-cap", "7"],
        ]
    else:
        if ns.suite == "phase2a":
            experiments = [
                ["--model-name", "zipvoice_distill", "--num-steps", "4", "--t-shift", "0.9"],
                ["--model-name", "zipvoice_distill", "--num-steps", "4", "--solver-mode", "lux_anchor"],
                ["--model-name", "zipvoice_distill", "--num-steps", "4", "--prompt-duration-cap", "3"],
                ["--model-name", "zipvoice_distill", "--num-steps", "4", "--prompt-duration-cap", "5"],
            ]
        else:
            experiments = [
                ["--model-name", "zipvoice_distill", "--num-steps", "4"],
                ["--model-name", "zipvoice_distill", "--num-steps", "4", "--vocoder-mode", "lux48k"],
                ["--preset", "tracka_tuned_no48k"],
                ["--preset", "tracka_tuned"],
            ]

    for exp in experiments:
        if ns.voices:
            exp += ["--voices", ns.voices]
        if ns.prompt_ids:
            exp += ["--prompt-ids", ns.prompt_ids]
        run_one(exp)


if __name__ == "__main__":
    main()
