r"""
Run the direct Lux vs Inflect base comparison set.

Examples:
  .\.venv-voxcpm\Scripts\python.exe scripts\run_lux_vs_inflect_bench.py
  .\.venv-voxcpm\Scripts\python.exe scripts\run_lux_vs_inflect_bench.py --voices jessica,henry --prompt-ids 06_emotional,08_clone_stress
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_one(script_name, args):
    cmd = [PYTHON, str(PROJECT_ROOT / "scripts" / script_name), *args]
    print("\n=== Running ===")
    print(" ".join(cmd))
    print()
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default=None)
    ap.add_argument("--prompt-ids", default=None)
    ns = ap.parse_args()

    common = []
    if ns.voices:
        common += ["--voices", ns.voices]
    if ns.prompt_ids:
        common += ["--prompt-ids", ns.prompt_ids]

    run_one("run_inflect_base_bench.py", common)
    run_one("test_true_lux.py", common)


if __name__ == "__main__":
    main()
