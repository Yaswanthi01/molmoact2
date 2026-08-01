#!/usr/bin/env python3
"""Evaluate multiple checkpoints and produce a comparison report.

This script is intended for a directory of checkpoint folders such as:
    checkpoints/panda-ee-50k-milestones/

It will:
1. Discover checkpoint directories matching step* under the provided root.
2. Run the open-loop evaluation for each checkpoint over a chosen episode range.
3. Aggregate per-checkpoint summaries.
4. Produce a comparison CSV/JSON and a simple plot.

It is designed to be a convenience wrapper around the existing evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple checkpoints via open-loop eval.")
    parser.add_argument("--checkpoint-root", required=True, help="Root directory containing checkpoint folders, e.g. checkpoints/panda-ee-50k-milestones")
    parser.add_argument("--dataset", required=True, help="LeRobot dataset repo id")
    parser.add_argument("--dataset-root", default=None, help="Optional local dataset root")
    parser.add_argument("--norm-tag", default="", help="MolmoAct2 normalization tag")
    parser.add_argument("--episode-start", type=int, default=0, help="First episode index")
    parser.add_argument("--episode-end", type=int, default=4, help="Last episode index (inclusive)")
    parser.add_argument("--expected-episodes", type=int, default=None, help="Expected number of episodes for aggregation")
    parser.add_argument("--output-root", default=None, help="Directory for all comparison outputs")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--video-backend", default="pyav", help="LeRobot video backend")
    parser.add_argument("--skip-eval", action="store_true", help="Only build comparison plots from existing outputs")
    return parser.parse_args()


def discover_checkpoints(root: Path) -> List[Path]:
    if not root.exists():
        raise SystemExit(f"Checkpoint root does not exist: {root}")
    checkpoints = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("step")],
        key=lambda p: int(p.name[4:]) if p.name[4:].isdigit() else p.name,
    )
    if not checkpoints:
        raise SystemExit(f"No checkpoint folders found under: {root}")
    return checkpoints


def run_eval(checkpoint: Path, args: argparse.Namespace, output_root: Path) -> Path:
    eval_root = output_root / checkpoint.name
    eval_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "experiments/scripts/run_open_loop_inference_lerobot.py",
        "--dataset",
        args.dataset,
        "--dataset_root",
        str(Path(args.dataset_root).expanduser()) if args.dataset_root else str(Path("datasets") / args.dataset),
        "--episode_idx",
        str(args.episode_start),
        "--checkpoint",
        str(checkpoint),
        "--output_dir",
        str(eval_root / f"episode{args.episode_start:03d}"),
        "--policy_type",
        "molmoact2",
        "--norm_tag",
        args.norm_tag or "",
        "--video_backend",
        args.video_backend,
        "--device",
        args.device,
    ]

    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    return eval_root


def collect_summary(path: Path) -> Dict[str, Any]:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.json at {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_comparison_report(results: List[Dict[str, Any]], output_root: Path) -> Dict[str, Any]:
    metrics = []
    for item in results:
        summary = item["summary"]
        metrics.append(
            {
                "checkpoint": item["checkpoint"],
                "overall_mse": summary.get("average_mse_raw"),
                "gripper_mse": summary.get("average_mse_raw_gripper"),
                "nongripper_mse": summary.get("average_mse_raw_nongripper"),
                "trajectory_len": summary.get("trajectory_len"),
                "episode_idx": summary.get("episode_idx"),
            }
        )

    report = {"comparison_type": "checkpoint_open_loop_eval", "checkpoints": metrics}
    output_path = output_root / "checkpoint_comparison.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def plot_comparison(report: Dict[str, Any], output_root: Path) -> Path:
    checkpoints = [item["checkpoint"] for item in report["checkpoints"]]
    overall = [float(item["overall_mse"]) if item["overall_mse"] is not None else np.nan for item in report["checkpoints"]]
    gripper = [float(item["gripper_mse"]) if item["gripper_mse"] is not None else np.nan for item in report["checkpoints"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(checkpoints))
    ax.bar(x - 0.2, overall, width=0.35, label="overall raw MSE")
    ax.bar(x + 0.2, gripper, width=0.35, label="gripper raw MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, rotation=30, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title("Checkpoint comparison")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path = output_root / "checkpoint_comparison.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else repo_root / "evaluations" / "checkpoint_comparison"
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(checkpoint_root)
    print(f"Found {len(checkpoints)} checkpoints under {checkpoint_root}")

    results: List[Dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_output = output_root / checkpoint.name
        if not args.skip_eval:
            eval_dir = run_eval(checkpoint, args, checkpoint_output)
            summary = collect_summary(eval_dir / f"episode{args.episode_start:03d}")
        else:
            summary = collect_summary(checkpoint_output / f"episode{args.episode_start:03d}")
        results.append({"checkpoint": checkpoint.name, "summary": summary})

    report = build_comparison_report(results, output_root)
    plot_path = plot_comparison(report, output_root)
    print(f"Wrote comparison JSON: {output_root / 'checkpoint_comparison.json'}")
    print(f"Wrote comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
