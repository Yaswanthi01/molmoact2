#!/usr/bin/env python3
"""Build one final report from aggregate action and binary gripper metrics.

This script only reads existing JSON reports. It does not load the model,
dataset, images, or videos and does not run inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge full open-loop and binary gripper evaluation reports."
    )
    parser.add_argument(
        "evaluation_root",
        type=Path,
        help="Evaluation directory containing the two input JSON reports.",
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=None,
        help="Default: EVALUATION_ROOT/full_evaluation_summary.json",
    )
    parser.add_argument(
        "--gripper-report",
        type=Path,
        default=None,
        help="Default: EVALUATION_ROOT/gripper_binary_summary.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Default: EVALUATION_ROOT/final_evaluation_report.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Default: EVALUATION_ROOT/final_evaluation_report.md",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Required report not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return value


def require_number(mapping: dict[str, Any], key: str, source: Path) -> float:
    value = mapping.get(key)
    if value is None:
        raise SystemExit(f"{source} has no numeric {key!r}")
    value = float(value)
    if not math.isfinite(value):
        raise SystemExit(f"{source} has non-finite {key}={value}")
    return value


def mean(values: list[float]) -> float:
    if not values:
        raise SystemExit("Cannot calculate a grouped MSE from an empty dimension list")
    return sum(values) / len(values)


def rmse(mse: float) -> float:
    if mse < 0:
        raise SystemExit(f"MSE cannot be negative: {mse}")
    return math.sqrt(mse)


def infer_action_groups(names: list[str]) -> dict[str, list[int]]:
    """Infer useful groups without requiring an embodiment-specific schema."""
    lowered = [name.lower() for name in names]
    groups: dict[str, list[int]] = {
        "gripper": [idx for idx, name in enumerate(lowered) if "gripper" in name],
        "rotation": [idx for idx, name in enumerate(lowered) if "rot" in name or "quat" in name],
    }
    position_tokens = {"x", "y", "z", "ee_x", "ee_y", "ee_z", "pos_x", "pos_y", "pos_z"}
    groups["position"] = [idx for idx, name in enumerate(lowered) if name in position_tokens]
    groups["non_gripper"] = [idx for idx in range(len(names)) if idx not in groups["gripper"]]
    return {name: indices for name, indices in groups.items() if indices}


def format_metric(value: Any, digits: int = 8) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.expanduser().resolve()
    aggregate_path = (
        args.aggregate_report.expanduser().resolve()
        if args.aggregate_report
        else root / "full_evaluation_summary.json"
    )
    gripper_path = (
        args.gripper_report.expanduser().resolve()
        if args.gripper_report
        else root / "gripper_binary_summary.json"
    )
    json_output = (
        args.json_output.expanduser().resolve()
        if args.json_output
        else root / "final_evaluation_report.json"
    )
    markdown_output = (
        args.markdown_output.expanduser().resolve()
        if args.markdown_output
        else root / "final_evaluation_report.md"
    )

    aggregate = read_json(aggregate_path)
    gripper = read_json(gripper_path)
    metrics = aggregate.get("metrics")
    binary_metrics = gripper.get("metrics")
    if not isinstance(metrics, dict):
        raise SystemExit(f"{aggregate_path} has no metrics object")
    if not isinstance(binary_metrics, dict):
        raise SystemExit(f"{gripper_path} has no metrics object")

    aggregate_frames = int(aggregate.get("total_evaluated_frames", 0))
    gripper_frames = int(gripper.get("total_frames", 0))
    if aggregate_frames <= 0 or gripper_frames <= 0:
        raise SystemExit("Both reports must contain a positive evaluated frame count")
    if aggregate_frames != gripper_frames:
        raise SystemExit(
            f"Frame-count mismatch: aggregate={aggregate_frames}, "
            f"gripper={gripper_frames}"
        )
    if int(aggregate.get("evaluated_episodes", 0)) != int(
        gripper.get("evaluated_episodes", 0)
    ):
        raise SystemExit("Episode-count mismatch between aggregate and gripper reports")

    action_names = [str(name) for name in aggregate.get("action_names", [])]
    raw_per_dim = [
        float(value) for value in metrics.get("average_mse_raw_per_dim", [])
    ]
    if not action_names or len(action_names) != len(raw_per_dim):
        raise SystemExit(
            "action_names and average_mse_raw_per_dim must have the same non-zero length"
        )
    if not all(math.isfinite(value) and value >= 0 for value in raw_per_dim):
        raise SystemExit("Raw per-dimension MSE contains invalid values")

    action_groups = infer_action_groups(action_names)
    grouped_metrics = {
        group: {
            "dimensions": [action_names[idx] for idx in indices],
            "mse": mean([raw_per_dim[idx] for idx in indices]),
            "rmse": rmse(mean([raw_per_dim[idx] for idx in indices])),
        }
        for group, indices in action_groups.items()
    }

    overall_raw_mse = require_number(metrics, "average_mse_raw", aggregate_path)
    ee_pose_raw_mse = metrics.get("average_mse_raw_nongripper")
    gripper_raw_mse = metrics.get("average_mse_raw_gripper")
    policy_scale_mse = require_number(
        metrics, "average_mse_policy_scale", aggregate_path
    )
    inferred_gripper_mse = grouped_metrics.get("gripper", {}).get("mse")
    if gripper_raw_mse is not None and inferred_gripper_mse is not None and not math.isclose(
        inferred_gripper_mse, float(gripper_raw_mse), rel_tol=1e-6, abs_tol=1e-10
    ):
        raise SystemExit(
            "Gripper MSE does not match the gripper entry in raw per-dimension MSE"
        )

    squared_error_sum_across_dims = sum(raw_per_dim)
    gripper_error_contribution = (
        None
        if squared_error_sum_across_dims == 0
        else (0.0 if gripper_raw_mse is None else float(gripper_raw_mse) / squared_error_sum_across_dims)
    )

    per_dimension = {
        name: {"mse": value, "rmse": rmse(value)}
        for name, value in zip(action_names, raw_per_dim)
    }

    final_report = {
        "report_type": "final_open_loop_evaluation",
        "evaluation_root": str(root),
        "source_reports": {
            "aggregate_action_report": str(aggregate_path),
            "binary_gripper_report": str(gripper_path),
        },
        "checkpoint": aggregate.get("checkpoint"),
        "dataset": aggregate.get("dataset"),
        "evaluation_complete": bool(aggregate.get("complete"))
        and bool(gripper.get("complete")),
        "evaluated_episodes": int(aggregate["evaluated_episodes"]),
        "total_evaluated_frames": aggregate_frames,
        "action_names": action_names,
        "primary_metrics": {
            "overall_raw_action_mse": overall_raw_mse,
            "overall_raw_action_rmse": rmse(overall_raw_mse),
            "ee_pose_raw_mse": ee_pose_raw_mse,
            "ee_pose_raw_rmse": None if ee_pose_raw_mse is None else rmse(float(ee_pose_raw_mse)),
            "ee_position_raw_mse": grouped_metrics.get("position", {}).get("mse"),
            "ee_position_raw_rmse": grouped_metrics.get("position", {}).get("rmse"),
            "ee_rotation_component_raw_mse": grouped_metrics.get("rotation", {}).get("mse"),
            "ee_rotation_component_raw_rmse": grouped_metrics.get("rotation", {}).get("rmse"),
            "gripper_raw_mse": gripper_raw_mse,
            "gripper_raw_rmse": None if gripper_raw_mse is None else rmse(float(gripper_raw_mse)),
            "gripper_binary_accuracy": binary_metrics.get("accuracy"),
            "gripper_binary_precision": binary_metrics.get("precision_positive"),
            "gripper_binary_recall": binary_metrics.get("recall_positive"),
            "gripper_binary_f1": binary_metrics.get("f1_positive"),
            "gripper_binary_specificity": binary_metrics.get(
                "specificity_negative"
            ),
        },
        "diagnostics": {
            "policy_scale_mse": policy_scale_mse,
            "policy_scale_warning": (
                "Normalization can differ by action dimension. Use raw grouped metrics "
                "for physical-scale interpretation and binary metrics for discrete grippers."
            ),
            "gripper_fraction_of_summed_raw_dimension_mse": (
                gripper_error_contribution
            ),
            "gripper_percentage_of_summed_raw_dimension_mse": (
                None
                if gripper_error_contribution is None
                else 100.0 * gripper_error_contribution
            ),
            "action_groups": grouped_metrics,
        },
        "raw_per_dimension": per_dimension,
        "gripper_binary": {
            "threshold": gripper.get("binary_threshold"),
            "ground_truth_lower_value": gripper.get("ground_truth_lower_value"),
            "ground_truth_upper_value": gripper.get("ground_truth_upper_value"),
            "class_counts": gripper.get("class_counts"),
            "confusion_matrix": gripper.get("confusion_matrix"),
            "metrics": binary_metrics,
        },
    }

    primary = final_report["primary_metrics"]
    diagnostic = final_report["diagnostics"]
    markdown = f"""# Final Open-Loop Evaluation

## Evaluation coverage

- Checkpoint: `{final_report["checkpoint"]}`
- Dataset: `{final_report["dataset"]}`
- Episodes: {final_report["evaluated_episodes"]}
- Frames: {final_report["total_evaluated_frames"]}
- Complete: {final_report["evaluation_complete"]}

## Primary metrics

| Metric | Value |
|---|---:|
| Overall raw action MSE | {format_metric(primary["overall_raw_action_mse"])} |
| Overall raw action RMSE | {format_metric(primary["overall_raw_action_rmse"])} |
| EE pose raw MSE | {format_metric(primary["ee_pose_raw_mse"])} |
| EE pose raw RMSE | {format_metric(primary["ee_pose_raw_rmse"])} |
| EE position raw MSE | {format_metric(primary["ee_position_raw_mse"])} |
| EE position raw RMSE | {format_metric(primary["ee_position_raw_rmse"])} |
| EE rotation-component raw MSE | {format_metric(primary["ee_rotation_component_raw_mse"])} |
| EE rotation-component raw RMSE | {format_metric(primary["ee_rotation_component_raw_rmse"])} |
| Gripper raw MSE | {format_metric(primary["gripper_raw_mse"])} |
| Gripper raw RMSE | {format_metric(primary["gripper_raw_rmse"])} |
| Gripper binary accuracy | {format_metric(primary["gripper_binary_accuracy"], 6)} |
| Gripper binary precision | {format_metric(primary["gripper_binary_precision"], 6)} |
| Gripper binary recall | {format_metric(primary["gripper_binary_recall"], 6)} |
| Gripper binary F1 | {format_metric(primary["gripper_binary_f1"], 6)} |
| Gripper binary specificity | {format_metric(primary["gripper_binary_specificity"], 6)} |

## Raw per-dimension metrics

| Action dimension | MSE | RMSE |
|---|---:|---:|
"""
    for name, values in per_dimension.items():
        markdown += (
            f"| {name} | {format_metric(values['mse'])} "
            f"| {format_metric(values['rmse'])} |\n"
        )

    markdown += f"""
## Gripper diagnostics

- Binary threshold: {format_metric(gripper.get("binary_threshold"), 6)}
- Gripper share of summed per-dimension raw MSE: {format_metric(diagnostic["gripper_percentage_of_summed_raw_dimension_mse"], 2)}%
- Confusion matrix: `{json.dumps(gripper.get("confusion_matrix"), sort_keys=True)}`

## Interpretation note

{diagnostic["policy_scale_warning"]}

This is open-loop action-prediction evaluation on recorded data. It does not
measure closed-loop task success or real-robot robustness.
"""

    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(markdown, encoding="utf-8")

    print(f"Episodes: {final_report['evaluated_episodes']}")
    print(f"Frames: {final_report['total_evaluated_frames']}")
    print(f"Overall raw action MSE: {overall_raw_mse}")
    print(f"EE pose raw MSE: {ee_pose_raw_mse}")
    print(f"Gripper binary accuracy: {binary_metrics.get('accuracy')}")
    print(f"Gripper binary F1: {binary_metrics.get('f1_positive')}")
    print(f"Wrote JSON: {json_output}")
    print(f"Wrote Markdown: {markdown_output}")


if __name__ == "__main__":
    main()
