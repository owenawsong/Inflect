r"""
Generate the focused Inflect base benchmark set.

Examples:
  .\.venv-voxcpm\Scripts\python.exe scripts\run_inflect_base_bench.py
  .\.venv-voxcpm\Scripts\python.exe scripts\run_inflect_base_bench.py --voices jessica,henry --prompt-ids 06_emotional,08_clone_stress
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
    ap.add_argument("--voices", default=None, help="Optional comma-separated benchmark voice subset.")
    ap.add_argument("--prompt-ids", default=None, help="Optional comma-separated benchmark prompt subset.")
    ns = ap.parse_args()

    experiments = [
        ["--model-name", "zipvoice_distill", "--num-steps", "4"],
        ["--model-name", "zipvoice_distill", "--num-steps", "4", "--vocoder-mode", "lux48k"],
        ["--preset", "inflect_base"],
        ["--preset", "inflect_base_solver"],
    ]

    for exp in experiments:
        if ns.voices:
            exp += ["--voices", ns.voices]
        if ns.prompt_ids:
            exp += ["--prompt-ids", ns.prompt_ids]
        run_one(exp)


if __name__ == "__main__":
    main()
