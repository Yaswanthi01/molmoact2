#!/usr/bin/env python3
"""Run all local validations for the Panda end-effector pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "datasets" / "panda_drawer_ee_molmoact2_v3"

PYTHON_FILES = (
    "experiments/launch_scripts/data_mixtures.py",
    "experiments/scripts/prepare_panda_drawer_dataset.py",
    "experiments/scripts/verify_panda_ee_actions.py",
    "experiments/scripts/validate_panda_videos.py",
    "experiments/scripts/preflight_panda_wrapper.py",
    "experiments/scripts/preflight_panda_model.py",
    "experiments/scripts/run_open_loop_inference_lerobot.py",
    "experiments/scripts/serve_policy.py",
)

SLURM_FILES = (
    "experiments/slurm/panda_smoke.slurm",
    "experiments/slurm/panda_pilot.slurm",
    "experiments/slurm/panda_full_1epoch.slurm",
    "experiments/slurm/panda_eval.slurm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip exhaustive MP4 decoding.",
    )
    return parser.parse_args()


def run(label: str, command: list[str]) -> None:
    print(f"\n{label}", flush=True)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()

    print("Repository:", REPO_ROOT)
    print("Python:", sys.executable)
    print("Dataset:", DATASET_ROOT)

    if not (DATASET_ROOT / "meta" / "info.json").is_file():
        raise SystemExit(
            f"Prepared EE dataset not found: {DATASET_ROOT}\n"
            "Run experiments/scripts/prepare_panda_drawer_dataset.py first."
        )

    run(
        "[1/6] Compiling Panda Python entry points...",
        [sys.executable, "-m", "py_compile", *PYTHON_FILES],
    )

    print("\n[2/6] Checking SLURM shell syntax...", flush=True)
    for slurm_file in SLURM_FILES:
        run(f"Checking {slurm_file}", ["bash", "-n", slurm_file])

    run(
        "[3/6] Verifying EE action concatenation...",
        [
            sys.executable,
            "experiments/scripts/verify_panda_ee_actions.py",
            "--dataset-root",
            str(DATASET_ROOT),
        ],
    )

    if args.quick:
        print("\n[4/6] Skipping full video decoding (--quick).", flush=True)
    else:
        run(
            "[4/6] Decoding and validating all videos...",
            [
                sys.executable,
                "experiments/scripts/validate_panda_videos.py",
                "--dataset-root",
                str(DATASET_ROOT),
            ],
        )

    run(
        "[5/6] Running LeRobot wrapper preflight...",
        [sys.executable, "experiments/scripts/preflight_panda_wrapper.py"],
    )
    run(
        "[6/6] Running MolmoAct2 model/collation preflight...",
        [sys.executable, "experiments/scripts/preflight_panda_model.py"],
    )

    print("\nPANDA EE PIPELINE VERIFICATION PASSED")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
