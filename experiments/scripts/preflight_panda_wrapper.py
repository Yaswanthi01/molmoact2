#!/usr/bin/env python3
"""Validate the Panda dataset through MolmoAct2's real LeRobot wrapper."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
CACHE_ROOT = REPO_ROOT / ".cache" / "panda_wrapper_preflight"
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "huggingface" / "datasets"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from launch_scripts.data_mixtures import TAG_METADATA_BY_TAG
from launch_scripts.lerobot_utils.stats import _collect_tagged_stats
from launch_scripts.lerobot_utils.train_plan import (
    _normalize_registered_lerobot_tag_metadata,
    get_lerobot_training_data_plan,
)
from olmo.data.lerobot_wrapper import build_lerobot_dataset


DATASET_REPO_ID = "panda_drawer_ee_molmoact2_v3"
DATASET_NAME = f"lerobot:{DATASET_REPO_ID}"
DATA_ROOT = REPO_ROOT / "datasets"


def _set_json_env(name: str, value: object) -> None:
    os.environ[name] = json.dumps(value)


def _validate_action_boundaries(wrapper: object, dataset_root: Path) -> None:
    parquet_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    frames = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["index", "episode_index", "frame_index"],
            )
            for path in parquet_paths
        ],
        ignore_index=True,
    )
    episode_groups = list(frames.groupby("episode_index", sort=True))
    selected_groups = [
        episode_groups[0],
        episode_groups[len(episode_groups) // 2],
        episode_groups[-1],
    ]

    print("\nValidating action chunks at episode boundaries...")
    checks = 0
    for episode_index, episode in selected_groups:
        episode = episode.sort_values("frame_index")
        episode_length = len(episode)
        for frames_from_end in [29, 1, 0]:
            row = episode.iloc[-1 - frames_from_end]
            dataset_index = int(row["index"])
            remaining_frames = frames_from_end + 1
            expected_pad = np.arange(30) >= remaining_frames

            raw = wrapper.dataset[dataset_index]
            raw_action = np.asarray(raw["action"])
            raw_pad = np.asarray(raw["action_is_pad"], dtype=np.bool_)
            assert raw_action.shape == (30, 8), (
                episode_index,
                frames_from_end,
                "raw action",
                raw_action.shape,
            )
            assert raw_pad.shape == (30,), (
                episode_index,
                frames_from_end,
                "raw pad",
                raw_pad.shape,
            )
            assert np.array_equal(raw_pad, expected_pad), (
                episode_index,
                frames_from_end,
                raw_pad,
                expected_pad,
            )

            example = wrapper[dataset_index]
            action = np.asarray(example["action"])
            horizon_pad = np.asarray(example["action_horizon_is_pad"], dtype=np.bool_)
            dimension_pad = np.asarray(example["action_dim_is_pad"], dtype=np.bool_)

            assert action.shape == (30, 8)
            assert np.isfinite(action).all()
            assert horizon_pad.shape == (30,) and not horizon_pad.any()
            assert dimension_pad.shape == (8,) and not dimension_pad.any()
            assert int(example["metadata"]["episode_index"]) == int(episode_index)

            if expected_pad.any():
                last_real_index = remaining_frames - 1
                assert np.allclose(
                    action[expected_pad],
                    action[last_real_index],
                    rtol=1e-5,
                    atol=1e-6,
                ), (
                    episode_index,
                    frames_from_end,
                    "episode padding did not repeat the final action",
                )

            checks += 1
            print(
                f"  episode={int(episode_index):3d} "
                f"frames_from_end={frames_from_end:2d} "
                f"raw_padded_steps={int(raw_pad.sum()):2d}"
            )

    print(f"ACTION BOUNDARY CHECK PASSED ({checks} boundary samples)")


def main() -> None:
    dataset_root = DATA_ROOT / DATASET_REPO_ID
    if not (dataset_root / "meta" / "info.json").is_file():
        raise SystemExit(f"Prepared dataset not found: {dataset_root}")

    plan = get_lerobot_training_data_plan("panda_drawer_ee")
    tag_metadata = _normalize_registered_lerobot_tag_metadata(TAG_METADATA_BY_TAG)
    ee_metadata = tag_metadata["panda_drawer_ee"]
    assert ee_metadata["action_key"] == "action"
    assert ee_metadata["state_keys"] == ["observation.state"]
    assert ee_metadata["control_mode"] == "absolute end-effector pose"
    assert ee_metadata["action_dim"] == 8
    assert ee_metadata["camera_keys"] == [
        "observation.images.gripper_cam.rgb",
        "observation.images.left_cam.left",
        "observation.images.right_cam.left",
    ]

    with (dataset_root / "meta" / "info.json").open("r", encoding="utf-8") as f:
        dataset_info = json.load(f)
    assert dataset_info["features"]["action"]["names"] == [
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_rot_0",
        "ee_rot_1",
        "ee_rot_2",
        "ee_rot_3",
        "gripper",
    ]
    stats_by_tag, repo_to_tag, default_tag = _collect_tagged_stats(
        plan.robot_mixture,
        root_base=str(DATA_ROOT),
        tag_metadata_by_tag=TAG_METADATA_BY_TAG,
    )

    os.environ["LEROBOT_DATA_ROOT"] = str(DATA_ROOT)
    os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"
    os.environ["LEROBOT_ACTION_FORMAT"] = "continuous"
    os.environ["LEROBOT_STATE_FORMAT"] = "continuous"
    os.environ["LEROBOT_NORM_MODE"] = "min_max"
    os.environ["LEROBOT_N_OBS_STEPS"] = "1"
    os.environ["LEROBOT_MAX_ACTION_HORIZON"] = "30"
    os.environ["LEROBOT_MAX_ACTION_DIM"] = "8"
    os.environ["LEROBOT_RANDOM_CAMERA_ORDER"] = "none"
    _set_json_env("LEROBOT_STYLE_SAMPLING_RATES", {"robot_action": 1.0})
    _set_json_env("LEROBOT_STATS_BY_TAG", stats_by_tag)
    _set_json_env("LEROBOT_REPO_TO_TAG", repo_to_tag)
    _set_json_env("LEROBOT_TAG_METADATA", tag_metadata)

    print("Dataset root:", dataset_root)
    print("Repository-to-tag mapping:", repo_to_tag)
    print("Default tag:", default_tag)

    wrapper = build_lerobot_dataset(
        DATASET_NAME,
        "train",
        n_obs_steps=1,
        max_action_horizon=30,
        max_action_dim=8,
    )
    print("Wrapper examples:", len(wrapper))

    for index in [0, len(wrapper) // 2, len(wrapper) - 1]:
        example = wrapper[index]
        state = np.asarray(example["state"])
        action = np.asarray(example["action"])
        images = example["image"]

        assert state.shape == (8,), (index, "state", state.shape)
        assert action.shape == (30, 8), (index, "action", action.shape)
        assert np.isfinite(state).all(), f"Non-finite state at wrapper index {index}"
        assert np.isfinite(action).all(), f"Non-finite action at wrapper index {index}"
        assert len(images) == 3, (index, "images", len(images))

        print(f"\nWrapper sample {index}")
        print("  keys:", sorted(example))
        print("  state:", state.shape, state.dtype)
        print("  action:", action.shape, action.dtype)
        print("  images:", len(images), [getattr(image, "size", None) for image in images])
        print("  metadata:", example["metadata"])

    _validate_action_boundaries(wrapper, dataset_root)
    print("\nMOLMOACT2 WRAPPER PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
