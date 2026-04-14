from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    variants = ",".join(
        [
            "zipvoice_distill_4step_cfg3p0",
            "zipvoice_distill_4step_cfg3p0_lux48k",
            "inflect_base",
            "inflect_base_solver",
            "lux_tts_real_normal_speed",
        ]
    )
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "serve_zipvoice_compare.py"),
        "--host",
        "127.0.0.1",
        "--port",
        "8813",
        "--only-variants",
        variants,
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
