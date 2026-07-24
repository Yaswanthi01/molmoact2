#!/usr/bin/env python3
"""Plot MolmoAct2 action-flow training loss from chained SLURM logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


STEP_PATTERN = re.compile(r"\[step=(\d+)/\d+")
LOSS_PATTERN = re.compile(
    r"train/action_flow_loss=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and plot train/action_flow_loss from SLURM .out logs."
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Directory containing chained SLURM output logs.",
    )
    parser.add_argument(
        "--pattern",
        default="molmoact2-panda-ee-full-1ep-*.out",
        help="Log filename glob pattern.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Rolling-mean window in logged points (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("action_flow_loss.pdf"),
        help="Output PDF or PNG path (default: action_flow_loss.pdf).",
    )
    return parser.parse_args()


def extract_losses(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    values_by_step: dict[int, float] = {}
    for path in paths:
        current_step: Optional[int] = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            step_match = STEP_PATTERN.search(line)
            if step_match:
                current_step = int(step_match.group(1))
                continue
            loss_match = LOSS_PATTERN.search(line)
            if loss_match and current_step is not None:
                values_by_step[current_step] = float(loss_match.group(1))

    if not values_by_step:
        raise SystemExit(
            "No train/action_flow_loss values found. Check the log directory and pattern."
        )

    ordered = sorted(values_by_step.items())
    return (
        np.asarray([step for step, _ in ordered], dtype=np.int64),
        np.asarray([loss for _, loss in ordered], dtype=np.float64),
    )


def rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, int]:
    resolved_window = min(max(1, window), len(values))
    if resolved_window == 1:
        return values.copy(), resolved_window
    kernel = np.ones(resolved_window, dtype=np.float64) / resolved_window
    return np.convolve(values, kernel, mode="valid"), resolved_window


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    paths = sorted(log_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No logs matched {log_dir / args.pattern}")

    steps, losses = extract_losses(paths)
    smoothed, window = rolling_mean(losses, args.window)
    smooth_steps = steps[window - 1 :]

    comparison_size = min(window, max(1, len(losses) // 4))
    start_mean = float(np.mean(losses[:comparison_size]))
    end_mean = float(np.mean(losses[-comparison_size:]))
    relative_change = (
        None if start_mean == 0 else 100.0 * (end_mean - start_mean) / abs(start_mean)
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.plot(
        steps,
        losses,
        color="#8fb9dd",
        alpha=0.45,
        linewidth=0.9,
        label="Logged action-flow loss",
    )
    axis.plot(
        smooth_steps,
        smoothed,
        color="#174a7e",
        linewidth=2.2,
        label=f"Rolling mean ({window} logged points)",
    )
    axis.set_title("MolmoAct2 action-expert training loss")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("train/action_flow_loss")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Log files read: {len(paths)}")
    print(f"Loss points: {len(losses)}")
    print(f"Step range: {int(steps[0])} to {int(steps[-1])}")
    print(f"First-window mean: {start_mean}")
    print(f"Last-window mean: {end_mean}")
    if relative_change is not None:
        print(f"First-to-last window change: {relative_change:.2f}%")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
