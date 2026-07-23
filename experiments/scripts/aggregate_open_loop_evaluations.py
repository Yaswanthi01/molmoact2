#!/usr/bin/env python3
"""Aggregate per-episode open-loop evaluation summaries.

Episode metrics are weighted by ``trajectory_len`` so that every evaluated
frame contributes equally, even when episodes have different lengths.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCALAR_METRICS = (
    "average_mse",
    "average_mse_policy_scale",
    "average_mse_raw",
    "average_mse_raw_gripper",
    "average_mse_raw_nongripper",
)

VECTOR_METRICS = (
    "average_mse_per_dim",
    "average_mse_policy_scale_per_dim",
    "average_mse_raw_per_dim",
)

CONSISTENCY_FIELDS = (
    "checkpoint",
    "dataset",
    "action_dim",
    "action_names",
    "norm_tag",
    "normalization_mode",
    "normalization_mask",
    "policy_type",
    "style",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one frame-weighted report from per-episode summary.json files."
    )
    parser.add_argument(
        "evaluation_root",
        type=Path,
        help="Directory containing episodeXXX/summary.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: EVALUATION_ROOT/full_evaluation_summary.json).",
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=101,
        help="Expected number of episodes (default: 101).",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Aggregate available summaries even if expected episodes are missing.",
    )
    return parser.parse_args()


def load_summaries(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    summary_paths = sorted(root.glob("episode*/summary.json"))
    if not summary_paths:
        raise SystemExit(f"No episode summary files found under: {root}")

    summaries: list[tuple[Path, dict[str, Any]]] = []
    for path in summary_paths:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Could not read {path}: {exc}") from exc
        summaries.append((path, summary))
    return summaries


def validate_summaries(
    summaries: list[tuple[Path, dict[str, Any]]],
    expected_episodes: int,
    allow_incomplete: bool,
) -> list[tuple[Path, dict[str, Any]]]:
    by_episode: dict[int, tuple[Path, dict[str, Any]]] = {}
    reference_path, reference = summaries[0]

    for path, summary in summaries:
        if "episode_idx" not in summary:
            raise SystemExit(f"{path} has no episode_idx")
        episode_idx = int(summary["episode_idx"])
        if episode_idx in by_episode:
            other_path = by_episode[episode_idx][0]
            raise SystemExit(
                f"Duplicate summary for episode {episode_idx}: {other_path} and {path}"
            )

        trajectory_len = int(summary.get("trajectory_len", 0))
        if trajectory_len <= 0:
            raise SystemExit(f"{path} has invalid trajectory_len={trajectory_len}")

        for field in CONSISTENCY_FIELDS:
            if summary.get(field) != reference.get(field):
                raise SystemExit(
                    f"Inconsistent {field!r}: {path} has {summary.get(field)!r}, "
                    f"but {reference_path} has {reference.get(field)!r}"
                )

        by_episode[episode_idx] = (path, summary)

    expected_ids = set(range(expected_episodes))
    found_ids = set(by_episode)
    missing_ids = sorted(expected_ids - found_ids)
    unexpected_ids = sorted(found_ids - expected_ids)

    if unexpected_ids:
        raise SystemExit(f"Unexpected episode indices: {unexpected_ids}")
    if missing_ids and not allow_incomplete:
        raise SystemExit(
            f"Missing {len(missing_ids)} episode summaries: {missing_ids}. "
            "Wait for the array to finish or pass --allow-incomplete."
        )

    return [by_episode[idx] for idx in sorted(by_episode)]


def weighted_scalar(
    summaries: list[tuple[Path, dict[str, Any]]], metric: str
) -> float | None:
    weighted_sum = 0.0
    total_weight = 0
    for path, summary in summaries:
        value = summary.get(metric)
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value):
            raise SystemExit(f"{path} has non-finite {metric}={value}")
        weight = int(summary["trajectory_len"])
        weighted_sum += value * weight
        total_weight += weight
    return None if total_weight == 0 else weighted_sum / total_weight


def weighted_vector(
    summaries: list[tuple[Path, dict[str, Any]]], metric: str
) -> list[float] | None:
    expected_dim: int | None = None
    weighted_sums: list[float] = []
    total_weight = 0

    for path, summary in summaries:
        values = summary.get(metric)
        if values is None:
            continue
        values = [float(value) for value in values]
        if not all(math.isfinite(value) for value in values):
            raise SystemExit(f"{path} has non-finite values in {metric}")
        if expected_dim is None:
            expected_dim = len(values)
            weighted_sums = [0.0] * expected_dim
        elif len(values) != expected_dim:
            raise SystemExit(
                f"{path} has {len(values)} values for {metric}; expected {expected_dim}"
            )

        weight = int(summary["trajectory_len"])
        for idx, value in enumerate(values):
            weighted_sums[idx] += value * weight
        total_weight += weight

    if total_weight == 0:
        return None
    return [value / total_weight for value in weighted_sums]


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "full_evaluation_summary.json"
    )

    summaries = validate_summaries(
        load_summaries(root),
        expected_episodes=args.expected_episodes,
        allow_incomplete=args.allow_incomplete,
    )
    first = summaries[0][1]
    episode_indices = [int(summary["episode_idx"]) for _, summary in summaries]
    total_frames = sum(int(summary["trajectory_len"]) for _, summary in summaries)
    missing_indices = sorted(set(range(args.expected_episodes)) - set(episode_indices))

    aggregate: dict[str, Any] = {
        "aggregation": "frame_weighted_mean_of_episode_metrics",
        "evaluation_root": str(root),
        "checkpoint": first.get("checkpoint"),
        "dataset": first.get("dataset"),
        "policy_type": first.get("policy_type"),
        "style": first.get("style"),
        "norm_tag": first.get("norm_tag"),
        "normalization_mode": first.get("normalization_mode"),
        "normalization_mask": first.get("normalization_mask"),
        "action_dim": first.get("action_dim"),
        "action_names": first.get("action_names"),
        "expected_episodes": args.expected_episodes,
        "evaluated_episodes": len(summaries),
        "evaluated_episode_indices": episode_indices,
        "missing_episode_indices": missing_indices,
        "complete": not missing_indices,
        "total_evaluated_frames": total_frames,
        "metrics": {},
        "episodes": [
            {
                "episode_idx": int(summary["episode_idx"]),
                "trajectory_len": int(summary["trajectory_len"]),
                "average_mse_raw": summary.get("average_mse_raw"),
                "average_mse_policy_scale": summary.get("average_mse_policy_scale"),
                "summary": str(path),
            }
            for path, summary in summaries
        ],
    }

    for metric in SCALAR_METRICS:
        aggregate["metrics"][metric] = weighted_scalar(summaries, metric)
    for metric in VECTOR_METRICS:
        aggregate["metrics"][metric] = weighted_vector(summaries, metric)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Episodes evaluated: {len(summaries)}/{args.expected_episodes}")
    print(f"Frames evaluated: {total_frames}")
    print(f"Complete: {aggregate['complete']}")
    print(f"Average raw MSE: {aggregate['metrics']['average_mse_raw']}")
    print(f"Average policy-scale MSE: {aggregate['metrics']['average_mse_policy_scale']}")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
