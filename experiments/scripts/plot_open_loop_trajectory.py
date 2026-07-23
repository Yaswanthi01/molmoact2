#!/usr/bin/env python3
"""Plot predicted and ground-truth actions for one evaluated episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a multi-panel action trajectory comparison plot."
    )
    parser.add_argument(
        "episode_dir",
        type=Path,
        help="Evaluation episode directory containing summary.json and .npy arrays.",
    )
    parser.add_argument(
        "--scale",
        choices=("raw", "policy_scale"),
        default="raw",
        help="Action scale to plot (default: raw).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .pdf or .png path (default: EPISODE_DIR/action_trajectory_comparison_SCALE.pdf).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_dir = args.episode_dir.expanduser().resolve()
    summary_path = episode_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Missing summary: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prediction_path = (
        episode_dir / f"predicted_action_trajectory_{args.scale}.npy"
    )
    ground_truth_path = (
        episode_dir / f"gt_action_trajectory_{args.scale}.npy"
    )
    if not prediction_path.is_file():
        raise SystemExit(f"Missing predictions: {prediction_path}")
    if not ground_truth_path.is_file():
        raise SystemExit(f"Missing ground truth: {ground_truth_path}")

    predicted = np.asarray(np.load(prediction_path), dtype=np.float64)
    ground_truth = np.asarray(np.load(ground_truth_path), dtype=np.float64)
    if predicted.ndim != 2 or ground_truth.ndim != 2:
        raise SystemExit(
            f"Expected 2D arrays, got pred={predicted.shape}, gt={ground_truth.shape}"
        )
    if predicted.shape != ground_truth.shape:
        raise SystemExit(
            f"Shape mismatch: pred={predicted.shape}, gt={ground_truth.shape}"
        )

    action_names = summary.get("action_names")
    if not isinstance(action_names, list) or len(action_names) != predicted.shape[1]:
        action_names = [f"action_{idx}" for idx in range(predicted.shape[1])]

    frame_indices = np.asarray(
        summary.get("frame_indices", list(range(predicted.shape[0]))),
        dtype=np.int64,
    )
    if frame_indices.shape != (predicted.shape[0],):
        frame_indices = np.arange(predicted.shape[0])

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else episode_dir / f"action_trajectory_comparison_{args.scale}.pdf"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        predicted.shape[1],
        1,
        figsize=(15, 2.0 * predicted.shape[1]),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for dim, (axis, name) in enumerate(zip(axes, action_names)):
        axis.plot(
            frame_indices,
            ground_truth[:, dim],
            color="black",
            linewidth=1.8,
            label="Ground truth",
        )
        axis.plot(
            frame_indices,
            predicted[:, dim],
            color="#ff7f0e",
            linewidth=1.6,
            label="Prediction",
        )
        axis.set_ylabel(str(name))
        axis.grid(alpha=0.2, linewidth=0.5)

    axes[0].legend(loc="best", frameon=False, ncol=2)
    axes[-1].set_xlabel("Episode frame")
    title = args.title or (
        f"MolmoAct2 open-loop trajectory — episode "
        f"{summary.get('episode_idx', '?')} ({args.scale})"
    )
    figure.suptitle(title, fontsize=14)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Frames plotted: {predicted.shape[0]}")
    print(f"Action dimensions: {predicted.shape[1]}")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
