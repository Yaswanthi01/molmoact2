#!/usr/bin/env python3
"""Evaluate binary gripper predictions from existing open-loop artifacts.

This script does not run the model. It reads the raw predicted and ground-truth
action arrays already saved in each ``episodeXXX`` evaluation directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PREDICTION_FILENAME = "predicted_action_trajectory_raw.npy"
GROUND_TRUTH_FILENAME = "gt_action_trajectory_raw.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute binary gripper metrics from existing episode evaluations."
    )
    parser.add_argument(
        "evaluation_root",
        type=Path,
        help="Directory containing episodeXXX/summary.json and raw action arrays.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: EVALUATION_ROOT/gripper_binary_summary.json).",
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=101,
        help="Expected number of episode summaries (default: 101).",
    )
    parser.add_argument(
        "--expected-episode-ids",
        default=None,
        help="Comma-separated episode IDs. Overrides the contiguous 0..N-1 expectation.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Evaluate available episodes even if expected episodes are missing.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Binary threshold in raw gripper units. By default, use the midpoint "
            "between the two ground-truth gripper values."
        ),
    )
    return parser.parse_args()


def safe_divide(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def parse_expected_ids(value: str | None, expected_episodes: int) -> set[int]:
    if value is None:
        if expected_episodes < 1:
            raise SystemExit("--expected-episodes must be positive")
        return set(range(expected_episodes))
    try:
        ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit("--expected-episode-ids must contain comma-separated integers") from exc
    if not ids or any(idx < 0 for idx in ids) or len(ids) != len(set(ids)):
        raise SystemExit("--expected-episode-ids must contain unique, non-negative IDs")
    return set(ids)


def find_gripper_dim(summary: dict[str, Any], path: Path) -> int:
    names = summary.get("action_names")
    if not isinstance(names, list):
        raise SystemExit(f"{path} has no action_names list")
    gripper_dims = [
        idx for idx, name in enumerate(names) if "gripper" in str(name).lower()
    ]
    if len(gripper_dims) != 1:
        raise SystemExit(
            f"{path} must contain exactly one gripper action dimension; "
            f"found {gripper_dims} in {names}"
        )
    return gripper_dims[0]


def load_episode_arrays(
    summary_path: Path,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {summary_path}: {exc}") from exc

    episode_idx = int(summary["episode_idx"])
    gripper_dim = find_gripper_dim(summary, summary_path)
    episode_dir = summary_path.parent
    prediction_path = episode_dir / PREDICTION_FILENAME
    ground_truth_path = episode_dir / GROUND_TRUTH_FILENAME

    if not prediction_path.is_file():
        raise SystemExit(f"Missing prediction array: {prediction_path}")
    if not ground_truth_path.is_file():
        raise SystemExit(f"Missing ground-truth array: {ground_truth_path}")

    predicted = np.asarray(np.load(prediction_path), dtype=np.float64)
    ground_truth = np.asarray(np.load(ground_truth_path), dtype=np.float64)
    if predicted.ndim != 2 or ground_truth.ndim != 2:
        raise SystemExit(
            f"Episode {episode_idx} arrays must be 2D; "
            f"pred={predicted.shape}, gt={ground_truth.shape}"
        )
    if predicted.shape != ground_truth.shape:
        raise SystemExit(
            f"Episode {episode_idx} shape mismatch: "
            f"pred={predicted.shape}, gt={ground_truth.shape}"
        )
    if gripper_dim >= predicted.shape[1]:
        raise SystemExit(
            f"Episode {episode_idx} gripper dimension {gripper_dim} is outside "
            f"array shape {predicted.shape}"
        )

    return (
        episode_idx,
        gripper_dim,
        predicted[:, gripper_dim],
        ground_truth[:, gripper_dim],
    )


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "gripper_binary_summary.json"
    )

    summary_paths = sorted(root.glob("episode*/summary.json"))
    if not summary_paths:
        raise SystemExit(f"No episode summaries found under: {root}")

    episodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    gripper_dim: int | None = None
    for summary_path in summary_paths:
        episode_idx, current_dim, predicted, ground_truth = load_episode_arrays(
            summary_path
        )
        if episode_idx in episodes:
            raise SystemExit(f"Duplicate summary for episode {episode_idx}")
        if gripper_dim is None:
            gripper_dim = current_dim
        elif current_dim != gripper_dim:
            raise SystemExit(
                f"Inconsistent gripper dimension: episode {episode_idx} uses "
                f"{current_dim}, expected {gripper_dim}"
            )
        episodes[episode_idx] = (predicted, ground_truth)

    expected_ids = parse_expected_ids(args.expected_episode_ids, args.expected_episodes)
    found_ids = set(episodes)
    missing_ids = sorted(expected_ids - found_ids)
    unexpected_ids = sorted(found_ids - expected_ids)
    if unexpected_ids:
        raise SystemExit(f"Unexpected episode indices: {unexpected_ids}")
    if missing_ids and not args.allow_incomplete:
        raise SystemExit(
            f"Missing {len(missing_ids)} episodes: {missing_ids}. "
            "Wait for evaluation to finish or pass --allow-incomplete."
        )

    predicted_raw = np.concatenate(
        [episodes[idx][0] for idx in sorted(episodes)], axis=0
    )
    ground_truth_raw = np.concatenate(
        [episodes[idx][1] for idx in sorted(episodes)], axis=0
    )
    if not np.all(np.isfinite(predicted_raw)):
        raise SystemExit("Predicted gripper values contain NaN or infinity")
    if not np.all(np.isfinite(ground_truth_raw)):
        raise SystemExit("Ground-truth gripper values contain NaN or infinity")

    unique_gt = np.unique(ground_truth_raw)
    if unique_gt.size != 2:
        raise SystemExit(
            "Expected exactly two raw ground-truth gripper values, but found "
            f"{unique_gt.size}: {unique_gt.tolist()}"
        )

    lower_value = float(unique_gt[0])
    upper_value = float(unique_gt[1])
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else (lower_value + upper_value) / 2.0
    )
    if not math.isfinite(threshold):
        raise SystemExit(f"Threshold must be finite, got {threshold}")

    ground_truth_binary = ground_truth_raw > threshold
    predicted_binary = predicted_raw > threshold

    true_negative = int(np.sum(~predicted_binary & ~ground_truth_binary))
    false_positive = int(np.sum(predicted_binary & ~ground_truth_binary))
    false_negative = int(np.sum(~predicted_binary & ground_truth_binary))
    true_positive = int(np.sum(predicted_binary & ground_truth_binary))
    total = int(ground_truth_binary.size)
    correct = true_positive + true_negative

    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    specificity = safe_divide(true_negative, true_negative + false_positive)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )

    report = {
        "evaluation_root": str(root),
        "expected_episodes": len(expected_ids),
        "expected_episode_indices": sorted(expected_ids),
        "evaluated_episodes": len(episodes),
        "evaluated_episode_indices": sorted(episodes),
        "missing_episode_indices": missing_ids,
        "complete": not missing_ids,
        "total_frames": total,
        "gripper_dimension": gripper_dim,
        "ground_truth_lower_value": lower_value,
        "ground_truth_upper_value": upper_value,
        "binary_threshold": threshold,
        "positive_class": f"raw_gripper > {threshold}",
        "negative_class": f"raw_gripper <= {threshold}",
        "class_counts": {
            "negative": int(np.sum(~ground_truth_binary)),
            "positive": int(np.sum(ground_truth_binary)),
        },
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "metrics": {
            "accuracy": safe_divide(correct, total),
            "precision_positive": precision,
            "recall_positive": recall,
            "specificity_negative": specificity,
            "f1_positive": f1,
            "raw_mse": float(np.mean(np.square(predicted_raw - ground_truth_raw))),
            "raw_mae": float(np.mean(np.abs(predicted_raw - ground_truth_raw))),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_output.replace(output)

    print(f"Episodes evaluated: {len(episodes)}/{len(expected_ids)}")
    print(f"Frames evaluated: {total}")
    print(f"Ground-truth values: {lower_value}, {upper_value}")
    print(f"Binary threshold: {threshold}")
    print(f"Accuracy: {report['metrics']['accuracy']}")
    print(f"F1 (upper-value class): {f1}")
    print(f"Raw gripper MSE: {report['metrics']['raw_mse']}")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
