#!/usr/bin/env python3
"""Prepare Panda drawer datasets with explicit EE pose in observation.state.

This creates non-overwriting MolmoAct2-ready LeRobot datasets for two
experiments:

1. absolute: input has current EE pose, action remains absolute EE pose.
2. relative: input has current EE pose, action is local EE delta pose.

Both variants keep action shape (8,):

    absolute = ee_pos(3) + ee_quat_wxyz(4) + gripper(1)
    relative = delta_ee_pos(3) + delta_ee_quat_wxyz(4) + gripper(1)

For the relative variant, each row's delta is computed from that row's current
observation.proprio.ee_* to that row's action.ee_*. This matches step-local
delta execution at inference time.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "datasets" / "panda_drawer_many_ee_fullres_v3.0"

VARIANT_OUTPUTS = {
    "absolute": REPO_ROOT / "datasets" / "panda_drawer_ee_state_absolute_molmoact2_v3",
    "relative": REPO_ROOT / "datasets" / "panda_drawer_ee_state_relative_molmoact2_v3",
}

STATE_KEYS = (
    "observation.proprio.joint_pos",
    "observation.proprio.gripper",
    "observation.proprio.ee_pos",
    "observation.proprio.ee_rot",
)
ABSOLUTE_ACTION_KEYS = ("action.ee_pos", "action.ee_rot", "action.gripper")
OUTPUT_STATE_KEY = "observation.state"
OUTPUT_ACTION_KEY = "action"

STATE_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_rot_0",
    "ee_rot_1",
    "ee_rot_2",
    "ee_rot_3",
]
ABSOLUTE_ACTION_NAMES = [
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_rot_0",
    "ee_rot_1",
    "ee_rot_2",
    "ee_rot_3",
    "gripper",
]
RELATIVE_ACTION_NAMES = [
    "delta_ee_x",
    "delta_ee_y",
    "delta_ee_z",
    "delta_ee_rot_0",
    "delta_ee_rot_1",
    "delta_ee_rot_2",
    "delta_ee_rot_3",
    "gripper",
]


def _require_dependencies() -> None:
    missing: list[str] = []
    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError:
        missing.append("pandas")
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError:
        missing.append("pyarrow")
    if missing:
        raise SystemExit(
            "Missing required package(s): "
            + ", ".join(missing)
            + ". Install them in the training environment, for example: "
            "`python3 -m pip install pandas pyarrow`."
        )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def _as_vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _stack_column(df: Any, key: str) -> np.ndarray:
    return np.stack([_as_vector(value) for value in df[key].to_numpy()], axis=0)


def _concat_columns(df: Any, keys: Iterable[str]) -> np.ndarray:
    return np.concatenate([_stack_column(df, key) for key in keys], axis=1).astype(np.float32)


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1e-8)


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    ).astype(np.float32)


def _relative_quat_wxyz(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    current = _normalize_quat(current.astype(np.float32))
    target = _normalize_quat(target.astype(np.float32))
    same_hemisphere = np.sum(current * target, axis=-1, keepdims=True) >= 0.0
    target = np.where(same_hemisphere, target, -target)
    return _normalize_quat(_quat_multiply_wxyz(_quat_conjugate_wxyz(current), target))


def _copy_dataset_tree(source: Path, output: Path, *, overwrite: bool, symlink_videos: bool) -> None:
    if output.exists():
        if not overwrite:
            raise SystemExit(
                f"Output already exists: {output}\n"
                "Pass --overwrite to replace it, or choose a different --output."
            )
        shutil.rmtree(output)

    shutil.copytree(source, output, ignore=shutil.ignore_patterns(".cache"))

    if symlink_videos:
        videos = output / "videos"
        if videos.exists() or videos.is_symlink():
            if videos.is_dir() and not videos.is_symlink():
                shutil.rmtree(videos)
            else:
                videos.unlink()
        videos.symlink_to(source.resolve() / "videos", target_is_directory=True)


def _add_columns(output: Path, *, variant: str) -> int:
    import pandas as pd

    parquet_paths = sorted((output / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No parquet files found under {output / 'data'}")

    required = set(STATE_KEYS).union(ABSOLUTE_ACTION_KEYS)
    total_rows = 0
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        missing = sorted(required.difference(df.columns))
        if missing:
            raise SystemExit(f"{parquet_path} is missing required columns: {missing}")

        state = _concat_columns(df, STATE_KEYS)
        if state.shape[1] != 15:
            raise SystemExit(f"{parquet_path}: expected 15D state, got {state.shape}")

        if variant == "absolute":
            action = _concat_columns(df, ABSOLUTE_ACTION_KEYS)
        elif variant == "relative":
            current_pos = _stack_column(df, "observation.proprio.ee_pos")
            target_pos = _stack_column(df, "action.ee_pos")
            current_rot = _stack_column(df, "observation.proprio.ee_rot")
            target_rot = _stack_column(df, "action.ee_rot")
            gripper = _stack_column(df, "action.gripper")
            delta_pos = (target_pos - current_pos).astype(np.float32)
            delta_rot = _relative_quat_wxyz(current_rot, target_rot)
            action = np.concatenate([delta_pos, delta_rot, gripper], axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        if action.shape[1] != 8:
            raise SystemExit(f"{parquet_path}: expected 8D action, got {action.shape}")

        df[OUTPUT_STATE_KEY] = list(state)
        df[OUTPUT_ACTION_KEY] = list(action)
        df.to_parquet(parquet_path, index=False)
        total_rows += len(df)

    return total_rows


def _vector_stats(values: np.ndarray, names: list[str]) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": int(values.shape[0]),
        "names": names,
    }


def _compute_output_stats(output: Path, *, variant: str) -> dict[str, dict[str, Any]]:
    import pandas as pd

    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    for parquet_path in sorted((output / "data").glob("*/*.parquet")):
        df = pd.read_parquet(parquet_path, columns=[OUTPUT_STATE_KEY, OUTPUT_ACTION_KEY])
        state_chunks.append(_stack_column(df, OUTPUT_STATE_KEY))
        action_chunks.append(_stack_column(df, OUTPUT_ACTION_KEY))

    state_values = np.concatenate(state_chunks, axis=0)
    action_values = np.concatenate(action_chunks, axis=0)
    action_names = ABSOLUTE_ACTION_NAMES if variant == "absolute" else RELATIVE_ACTION_NAMES
    return {
        OUTPUT_STATE_KEY: _vector_stats(state_values, STATE_NAMES),
        OUTPUT_ACTION_KEY: _vector_stats(action_values, action_names),
    }


def _update_metadata(output: Path, *, variant: str) -> None:
    info_path = output / "meta" / "info.json"
    stats_path = output / "meta" / "stats.json"

    info = _load_json(info_path)
    features = info["features"]
    action_names = ABSOLUTE_ACTION_NAMES if variant == "absolute" else RELATIVE_ACTION_NAMES

    features[OUTPUT_STATE_KEY] = {
        "shape": [15],
        "dtype": "float32",
        "names": STATE_NAMES,
    }
    features[OUTPUT_ACTION_KEY] = {
        "shape": [8],
        "dtype": "float32",
        "names": action_names,
    }
    for feature in features.values():
        if feature.get("dtype") == "video":
            feature["info"]["video.codec"] = "av1"
            shape = tuple(feature.get("shape", ()))
            if not feature.get("names") and len(shape) == 3:
                if shape[-1] in {1, 3, 4}:
                    feature["names"] = ["height", "width", "channels"]
                elif shape[0] in {1, 3, 4}:
                    feature["names"] = ["channels", "height", "width"]
    info["robot_type"] = info.get("robot_type") or "franka_panda"
    _write_json(info_path, info)

    stats = _load_json(stats_path)
    stats.update(_compute_output_stats(output, variant=variant))
    _write_json(stats_path, stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_OUTPUTS),
        required=True,
        help="Dataset variant to create.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source LeRobot v3.0 dataset root.")
    parser.add_argument("--output", type=Path, default=None, help="Prepared dataset output root.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it exists.")
    parser.add_argument(
        "--symlink-videos",
        action="store_true",
        help="Symlink videos from the source dataset instead of copying them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = (args.output or VARIANT_OUTPUTS[args.variant]).expanduser().resolve()

    if not source.exists():
        raise SystemExit(f"Source dataset does not exist: {source}")
    if not (source / "meta" / "info.json").exists():
        raise SystemExit(f"Source does not look like a LeRobot dataset: {source}")

    _require_dependencies()
    _copy_dataset_tree(source, output, overwrite=args.overwrite, symlink_videos=args.symlink_videos)
    rows = _add_columns(output, variant=args.variant)
    _update_metadata(output, variant=args.variant)

    print(f"Prepared {args.variant} dataset: {output}")
    print(f"Rows updated: {rows}")
    print(f"  {OUTPUT_STATE_KEY} = joint_pos + gripper + ee_pos + ee_rot")
    if args.variant == "absolute":
        print(f"  {OUTPUT_ACTION_KEY} = ee_pos + ee_rot + gripper")
    else:
        print(f"  {OUTPUT_ACTION_KEY} = delta_ee_pos + delta_ee_rot + gripper")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
