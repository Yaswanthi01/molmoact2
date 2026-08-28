#!/usr/bin/env python3
"""Compute enhanced open-loop metrics from saved prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evaluation_root",
        type=Path,
        help="Directory containing episodeXXX evaluation directories.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Chunk length override. By default, read n_action_steps from summaries.",
    )
    parser.add_argument(
        "--transition-window",
        type=int,
        default=2,
        help="Frames on either side of a ground-truth gripper transition.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: EVALUATION_ROOT/enhanced_open_loop_summary.json",
    )
    return parser.parse_args()


def describe(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "rmse": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Quaternion trajectory contains a zero-length quaternion")
    return quaternions / norms


def quaternion_angular_error_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = normalize_quaternions(first)
    second = normalize_quaternions(second)
    dots = np.abs(np.sum(first * second, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))


def find_dimensions(action_names: list[str]) -> tuple[list[int], list[int], int]:
    lowered = [name.lower() for name in action_names]
    position_names = {"ee_x", "ee_y", "ee_z", "pos_x", "pos_y", "pos_z"}
    position = [idx for idx, name in enumerate(lowered) if name in position_names]
    rotation = [
        idx for idx, name in enumerate(lowered) if "rot" in name or "quat" in name
    ]
    gripper = [idx for idx, name in enumerate(lowered) if "gripper" in name]
    if len(position) != 3:
        raise ValueError(f"Expected three position dimensions, found {position}")
    if len(rotation) != 4:
        raise ValueError(f"Expected four quaternion dimensions, found {rotation}")
    if len(gripper) != 1:
        raise ValueError(f"Expected one gripper dimension, found {gripper}")
    return position, rotation, gripper[0]


def load_episode(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episode_dir = summary_path.parent
    predicted = np.load(episode_dir / "predicted_action_trajectory_raw.npy")
    ground_truth = np.load(episode_dir / "gt_action_trajectory_raw.npy")
    if predicted.shape != ground_truth.shape or predicted.ndim != 2:
        raise ValueError(
            f"Invalid action shapes for {episode_dir}: {predicted.shape}, {ground_truth.shape}"
        )
    if predicted.shape[0] == 0:
        raise ValueError(f"Empty action trajectory: {episode_dir}")
    return {
        "summary": summary,
        "predicted": predicted.astype(np.float64),
        "ground_truth": ground_truth.astype(np.float64),
    }


def accuracy(predicted: np.ndarray, ground_truth: np.ndarray) -> float | None:
    if predicted.size == 0:
        return None
    return float(np.mean(predicted == ground_truth))


def equal_episode_mean(episodes: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for episode in episodes:
        value: Any = episode
        for key in path:
            value = value[key]
        if value is not None:
            values.append(float(value))
    return None if not values else float(np.mean(values))


def main() -> None:
    args = parse_args()
    if args.transition_window < 0:
        raise SystemExit("--transition-window must be non-negative")

    root = args.evaluation_root.expanduser().resolve()
    summary_paths = sorted(root.glob("episode*/summary.json"))
    if not summary_paths:
        raise SystemExit(f"No episode summaries found under: {root}")

    loaded = [load_episode(path) for path in summary_paths]
    first_summary = loaded[0]["summary"]
    action_names = [str(name) for name in first_summary["action_names"]]
    position_dims, rotation_dims, gripper_dim = find_dimensions(action_names)

    summary_chunk_lengths = {
        int(item["summary"]["n_action_steps"])
        for item in loaded
        if item["summary"].get("n_action_steps") is not None
    }
    if args.n_action_steps is not None:
        n_action_steps = int(args.n_action_steps)
    elif len(summary_chunk_lengths) == 1:
        n_action_steps = summary_chunk_lengths.pop()
    else:
        raise SystemExit(
            "Could not infer one n_action_steps value; pass --n-action-steps explicitly"
        )
    if n_action_steps < 1:
        raise SystemExit("n_action_steps must be positive")

    all_gt_gripper = np.concatenate(
        [item["ground_truth"][:, gripper_dim] for item in loaded]
    )
    gripper_values = np.unique(all_gt_gripper)
    if gripper_values.size != 2:
        raise SystemExit(
            f"Expected two ground-truth gripper values, found {gripper_values.tolist()}"
        )
    gripper_threshold = float(np.mean(gripper_values))

    episodes: list[dict[str, Any]] = []
    pooled: dict[str, list[np.ndarray]] = {
        "position_error": [],
        "rotation_error_deg": [],
        "pred_translation_jump": [],
        "gt_translation_jump": [],
        "pred_rotation_jump_deg": [],
        "gt_rotation_jump_deg": [],
        "pred_gripper_binary": [],
        "gt_gripper_binary": [],
        "near_transition": [],
        "at_transition": [],
    }
    chunk_values: dict[int, dict[str, list[np.ndarray]]] = {
        step: {
            "position_error": [],
            "rotation_error_deg": [],
            "pred_gripper_binary": [],
            "gt_gripper_binary": [],
        }
        for step in range(n_action_steps)
    }

    for item in loaded:
        summary = item["summary"]
        predicted = item["predicted"]
        ground_truth = item["ground_truth"]
        if [str(name) for name in summary["action_names"]] != action_names:
            raise ValueError(f"Inconsistent action names in episode {summary['episode_idx']}")

        predicted_position = predicted[:, position_dims]
        gt_position = ground_truth[:, position_dims]
        predicted_rotation = predicted[:, rotation_dims]
        gt_rotation = ground_truth[:, rotation_dims]
        position_error = np.linalg.norm(predicted_position - gt_position, axis=1)
        rotation_error = quaternion_angular_error_deg(
            predicted_rotation, gt_rotation
        )

        predicted_gripper = predicted[:, gripper_dim] > gripper_threshold
        gt_gripper = ground_truth[:, gripper_dim] > gripper_threshold
        transition = np.zeros(gt_gripper.shape[0], dtype=bool)
        transition[1:] = gt_gripper[1:] != gt_gripper[:-1]
        near_transition = transition.copy()
        transition_indices = np.flatnonzero(transition)
        for index in transition_indices:
            start = max(0, int(index) - args.transition_window)
            end = min(gt_gripper.shape[0], int(index) + args.transition_window + 1)
            near_transition[start:end] = True

        predicted_translation_jump = np.linalg.norm(
            np.diff(predicted_position, axis=0), axis=1
        )
        gt_translation_jump = np.linalg.norm(np.diff(gt_position, axis=0), axis=1)
        predicted_rotation_jump = quaternion_angular_error_deg(
            predicted_rotation[1:], predicted_rotation[:-1]
        )
        gt_rotation_jump = quaternion_angular_error_deg(
            gt_rotation[1:], gt_rotation[:-1]
        )

        episode_metrics = {
            "episode_idx": int(summary["episode_idx"]),
            "frames": int(predicted.shape[0]),
            "position_error_raw": describe(position_error),
            "quaternion_angular_error_deg": describe(rotation_error),
            "predicted_translation_jump_raw": describe(predicted_translation_jump),
            "ground_truth_translation_jump_raw": describe(gt_translation_jump),
            "predicted_rotation_jump_deg": describe(predicted_rotation_jump),
            "ground_truth_rotation_jump_deg": describe(gt_rotation_jump),
            "gripper_accuracy": accuracy(predicted_gripper, gt_gripper),
            "gripper_accuracy_near_transitions": accuracy(
                predicted_gripper[near_transition], gt_gripper[near_transition]
            ),
            "gripper_accuracy_away_from_transitions": accuracy(
                predicted_gripper[~near_transition], gt_gripper[~near_transition]
            ),
            "gripper_accuracy_at_transition": accuracy(
                predicted_gripper[transition], gt_gripper[transition]
            ),
            "gripper_transition_count": int(np.sum(transition)),
            "gripper_near_transition_frame_count": int(np.sum(near_transition)),
        }
        episodes.append(episode_metrics)

        pooled["position_error"].append(position_error)
        pooled["rotation_error_deg"].append(rotation_error)
        pooled["pred_translation_jump"].append(predicted_translation_jump)
        pooled["gt_translation_jump"].append(gt_translation_jump)
        pooled["pred_rotation_jump_deg"].append(predicted_rotation_jump)
        pooled["gt_rotation_jump_deg"].append(gt_rotation_jump)
        pooled["pred_gripper_binary"].append(predicted_gripper)
        pooled["gt_gripper_binary"].append(gt_gripper)
        pooled["near_transition"].append(near_transition)
        pooled["at_transition"].append(transition)

        phases = np.arange(predicted.shape[0]) % n_action_steps
        for step in range(n_action_steps):
            mask = phases == step
            chunk_values[step]["position_error"].append(position_error[mask])
            chunk_values[step]["rotation_error_deg"].append(rotation_error[mask])
            chunk_values[step]["pred_gripper_binary"].append(predicted_gripper[mask])
            chunk_values[step]["gt_gripper_binary"].append(gt_gripper[mask])

    predicted_gripper_all = np.concatenate(pooled["pred_gripper_binary"])
    gt_gripper_all = np.concatenate(pooled["gt_gripper_binary"])
    near_transition_all = np.concatenate(pooled["near_transition"])
    at_transition_all = np.concatenate(pooled["at_transition"])

    chunk_report: list[dict[str, Any]] = []
    for step in range(n_action_steps):
        values = chunk_values[step]
        predicted_gripper = np.concatenate(values["pred_gripper_binary"])
        gt_gripper = np.concatenate(values["gt_gripper_binary"])
        chunk_report.append(
            {
                "action_step": step + 1,
                "position_error_raw": describe(
                    np.concatenate(values["position_error"])
                ),
                "quaternion_angular_error_deg": describe(
                    np.concatenate(values["rotation_error_deg"])
                ),
                "gripper_accuracy": accuracy(predicted_gripper, gt_gripper),
            }
        )

    frame_weighted = {
        "position_error_raw": describe(np.concatenate(pooled["position_error"])),
        "quaternion_angular_error_deg": describe(
            np.concatenate(pooled["rotation_error_deg"])
        ),
        "predicted_translation_jump_raw": describe(
            np.concatenate(pooled["pred_translation_jump"])
        ),
        "ground_truth_translation_jump_raw": describe(
            np.concatenate(pooled["gt_translation_jump"])
        ),
        "predicted_rotation_jump_deg": describe(
            np.concatenate(pooled["pred_rotation_jump_deg"])
        ),
        "ground_truth_rotation_jump_deg": describe(
            np.concatenate(pooled["gt_rotation_jump_deg"])
        ),
        "gripper_accuracy": accuracy(predicted_gripper_all, gt_gripper_all),
        "gripper_accuracy_near_transitions": accuracy(
            predicted_gripper_all[near_transition_all],
            gt_gripper_all[near_transition_all],
        ),
        "gripper_accuracy_away_from_transitions": accuracy(
            predicted_gripper_all[~near_transition_all],
            gt_gripper_all[~near_transition_all],
        ),
        "gripper_accuracy_at_transition": accuracy(
            predicted_gripper_all[at_transition_all],
            gt_gripper_all[at_transition_all],
        ),
        "gripper_transition_count": int(np.sum(at_transition_all)),
        "gripper_near_transition_frame_count": int(np.sum(near_transition_all)),
    }
    equal_episode = {
        "position_error_raw_mean": equal_episode_mean(
            episodes, ("position_error_raw", "mean")
        ),
        "position_error_raw_rmse": equal_episode_mean(
            episodes, ("position_error_raw", "rmse")
        ),
        "quaternion_angular_error_deg_mean": equal_episode_mean(
            episodes, ("quaternion_angular_error_deg", "mean")
        ),
        "quaternion_angular_error_deg_rmse": equal_episode_mean(
            episodes, ("quaternion_angular_error_deg", "rmse")
        ),
        "predicted_translation_jump_raw_mean": equal_episode_mean(
            episodes, ("predicted_translation_jump_raw", "mean")
        ),
        "predicted_rotation_jump_deg_mean": equal_episode_mean(
            episodes, ("predicted_rotation_jump_deg", "mean")
        ),
        "gripper_accuracy": equal_episode_mean(episodes, ("gripper_accuracy",)),
        "gripper_accuracy_near_transitions": equal_episode_mean(
            episodes, ("gripper_accuracy_near_transitions",)
        ),
        "gripper_accuracy_away_from_transitions": equal_episode_mean(
            episodes, ("gripper_accuracy_away_from_transitions",)
        ),
        "gripper_accuracy_at_transition": equal_episode_mean(
            episodes, ("gripper_accuracy_at_transition",)
        ),
    }

    report = {
        "report_type": "enhanced_open_loop_artifact_analysis",
        "evaluation_root": str(root),
        "checkpoint": first_summary.get("checkpoint"),
        "dataset": first_summary.get("dataset"),
        "norm_tag": first_summary.get("norm_tag"),
        "evaluated_episode_indices": [episode["episode_idx"] for episode in episodes],
        "evaluated_episodes": len(episodes),
        "total_frames": int(sum(episode["frames"] for episode in episodes)),
        "action_names": action_names,
        "n_action_steps": n_action_steps,
        "gripper_threshold": gripper_threshold,
        "gripper_transition_window_frames": int(args.transition_window),
        "frame_weighted": frame_weighted,
        "equal_episode_weighted": equal_episode,
        "error_by_action_step_within_chunk": chunk_report,
        "episodes": episodes,
        "random_seed_analysis": {
            "available_seeds": sorted(
                {
                    int(item["summary"]["seed"])
                    for item in loaded
                    if item["summary"].get("seed") is not None
                }
            ),
            "multiple_seed_comparison_available": False,
            "requires_new_model_inference": True,
        },
        "notes": [
            "Quaternion angular errors are sign-invariant and reported in degrees.",
            "Jump metrics do not cross episode boundaries.",
            "Action-step positions are reconstructed from each reset action queue order.",
            "Multiple random seeds cannot be reconstructed from one saved prediction trajectory.",
        ],
    }

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "enhanced_open_loop_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Episodes analyzed: {len(episodes)}")
    print(f"Frames analyzed: {report['total_frames']}")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
