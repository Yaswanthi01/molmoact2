#!/usr/bin/env python3
"""Verify that prepared Panda actions are ee_pos + ee_rot + gripper."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "panda_drawer_ee_molmoact2_v3"
SOURCE_ACTION_KEYS = ("action.ee_pos", "action.ee_rot", "action.gripper")
OUTPUT_ACTION_KEY = "action"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Prepared Panda LeRobot dataset root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    parquet_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No parquet files found under {dataset_root / 'data'}")

    selected_paths = sorted({
        parquet_paths[0],
        parquet_paths[len(parquet_paths) // 2],
        parquet_paths[-1],
    })

    checks = 0
    for parquet_path in selected_paths:
        frame = pd.read_parquet(
            parquet_path,
            columns=[*SOURCE_ACTION_KEYS, OUTPUT_ACTION_KEY],
        )
        selected_rows = sorted({0, len(frame) // 2, len(frame) - 1})
        for row_index in selected_rows:
            row = frame.iloc[row_index]
            expected = np.concatenate(
                [np.asarray(row[key], dtype=np.float32).reshape(-1) for key in SOURCE_ACTION_KEYS]
            )
            actual = np.asarray(row[OUTPUT_ACTION_KEY], dtype=np.float32).reshape(-1)
            if actual.shape != (8,):
                raise AssertionError(
                    f"{parquet_path} row {row_index}: expected action shape (8,), got {actual.shape}"
                )
            if not np.allclose(actual, expected):
                raise AssertionError(
                    f"{parquet_path} row {row_index}: combined action does not equal "
                    "action.ee_pos + action.ee_rot + action.gripper"
                )
            checks += 1

    first = pd.read_parquet(
        selected_paths[0],
        columns=[*SOURCE_ACTION_KEYS, OUTPUT_ACTION_KEY],
    ).iloc[0]
    print("ee_pos:", np.asarray(first["action.ee_pos"]))
    print("ee_rot:", np.asarray(first["action.ee_rot"]))
    print("gripper:", np.asarray(first["action.gripper"]))
    print("combined action:", np.asarray(first[OUTPUT_ACTION_KEY]))
    print(f"EE ACTION CONCATENATION PASSED ({checks} sampled rows)")


if __name__ == "__main__":
    main()
