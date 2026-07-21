#!/usr/bin/env python3
"""Prepare the Panda drawer LeRobot v3.0 dataset for MolmoAct2 fine-tuning.

The source dataset has split robot columns. This pipeline uses end-effector
actions, as required by the Panda controller interface:

    observation.proprio.joint_pos, observation.proprio.gripper
    action.ee_pos, action.ee_rot, action.gripper

MolmoAct2's training mixture expects one state key and one action key, so this
script creates a local copy with:

    observation.state = concat(observation.proprio.joint_pos,
                               observation.proprio.gripper)
    action = concat(action.ee_pos, action.ee_rot, action.gripper)

The source dataset is left untouched.
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
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "panda_drawer_ee_molmoact2_v3"

STATE_KEYS = ("observation.proprio.joint_pos", "observation.proprio.gripper")
ACTION_KEYS = ("action.ee_pos", "action.ee_rot", "action.gripper")
OUTPUT_STATE_KEY = "observation.state"
OUTPUT_ACTION_KEY = "action"


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


def _concat_row(row: Any, keys: Iterable[str]) -> np.ndarray:
    return np.concatenate([_as_vector(row[key]) for key in keys], axis=0).astype(np.float32)


def _copy_dataset_tree(source: Path, output: Path, *, overwrite: bool, symlink_videos: bool) -> None:
    if output.exists():
        if not overwrite:
            raise SystemExit(
                f"Output already exists: {output}\n"
                "Pass --overwrite to replace it, or choose a different --output."
            )
        shutil.rmtree(output)

    ignore = shutil.ignore_patterns(".cache")
    shutil.copytree(source, output, ignore=ignore)

    if symlink_videos:
        videos = output / "videos"
        if videos.exists() or videos.is_symlink():
            if videos.is_dir() and not videos.is_symlink():
                shutil.rmtree(videos)
            else:
                videos.unlink()
        videos.symlink_to(source.resolve() / "videos", target_is_directory=True)


def _add_combined_columns(output: Path) -> int:
    import pandas as pd

    parquet_paths = sorted((output / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No parquet files found under {output / 'data'}")

    total_rows = 0
    required = {*STATE_KEYS, *ACTION_KEYS}
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        missing = sorted(required.difference(df.columns))
        if missing:
            raise SystemExit(f"{parquet_path} is missing required columns: {missing}")

        df[OUTPUT_STATE_KEY] = df.apply(lambda row: _concat_row(row, STATE_KEYS), axis=1)
        df[OUTPUT_ACTION_KEY] = df.apply(lambda row: _concat_row(row, ACTION_KEYS), axis=1)
        df.to_parquet(parquet_path, index=False)
        total_rows += len(df)

    return total_rows


def _combine_stats(keys: tuple[str, ...], stats: dict[str, Any]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    vector_stat_keys = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
    for stat_key in vector_stat_keys:
        values = []
        for key in keys:
            if stat_key not in stats[key]:
                values = []
                break
            values.append(np.asarray(stats[key][stat_key], dtype=np.float64).reshape(-1))
        if values:
            combined[stat_key] = np.concatenate(values, axis=0).tolist()

    counts = [stats[key].get("count") for key in keys if "count" in stats[key]]
    if counts:
        first_count = counts[0]
        combined["count"] = first_count

    return combined


def _update_metadata(output: Path) -> None:
    info_path = output / "meta" / "info.json"
    stats_path = output / "meta" / "stats.json"

    info = _load_json(info_path)
    features = info["features"]

    features[OUTPUT_STATE_KEY] = {
        "shape": [8],
        "dtype": "float32",
        "names": [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper",
        ],
    }
    features[OUTPUT_ACTION_KEY] = {
        "shape": [8],
        "dtype": "float32",
        "names": [
            "ee_x",
            "ee_y",
            "ee_z",
            "ee_rot_0",
            "ee_rot_1",
            "ee_rot_2",
            "ee_rot_3",
            "gripper",
        ],
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
    missing_stats = [key for key in (*STATE_KEYS, *ACTION_KEYS) if key not in stats]
    if missing_stats:
        raise SystemExit(f"{stats_path} is missing stats for: {missing_stats}")

    stats[OUTPUT_STATE_KEY] = _combine_stats(STATE_KEYS, stats)
    stats[OUTPUT_ACTION_KEY] = _combine_stats(ACTION_KEYS, stats)
    _write_json(stats_path, stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source LeRobot v3.0 dataset root.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Prepared dataset output root.")
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
    output = args.output.expanduser().resolve()

    if not source.exists():
        raise SystemExit(f"Source dataset does not exist: {source}")
    if not (source / "meta" / "info.json").exists():
        raise SystemExit(f"Source does not look like a LeRobot dataset: {source}")

    _require_dependencies()
    _copy_dataset_tree(source, output, overwrite=args.overwrite, symlink_videos=args.symlink_videos)
    rows = _add_combined_columns(output)
    _update_metadata(output)

    print(f"Prepared dataset: {output}")
    print(f"Rows updated: {rows}")
    print("Added keys:")
    print(f"  {OUTPUT_STATE_KEY} = {STATE_KEYS[0]} + {STATE_KEYS[1]}")
    print(f"  {OUTPUT_ACTION_KEY} = {' + '.join(ACTION_KEYS)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
