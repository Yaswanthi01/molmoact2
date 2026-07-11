#!/usr/bin/env python3
"""Validate MolmoAct2 configuration and preprocessing for the Panda dataset."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
DATA_ROOT = REPO_ROOT / "datasets"
DATASET_REPO_ID = "panda_drawer_molmoact2_v3"
CACHE_ROOT = REPO_ROOT / ".cache" / "panda_model_preflight"

os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "huggingface" / "datasets"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from launch_scripts.data_mixtures import TAG_METADATA_BY_TAG
from launch_scripts.lerobot_utils.stats import _collect_tagged_stats
from launch_scripts.lerobot_utils.hf import get_hf_model_config
from launch_scripts.lerobot_utils.train_plan import (
    _normalize_registered_lerobot_tag_metadata,
    get_lerobot_training_data_plan,
)
from olmo.data.dataset import DeterministicDataset
from olmo.data.lerobot_wrapper import build_lerobot_dataset
from olmo.data.robot_processing import RobotProcessorConfig
from olmo.models.molmoact2.molmoact2 import MolmoAct2Config


def _set_json_env(name: str, value: object) -> None:
    os.environ[name] = json.dumps(value)


def _shape(value: object) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(int(v) for v in shape)


def main() -> None:
    dataset_root = DATA_ROOT / DATASET_REPO_ID
    if not (dataset_root / "meta" / "info.json").is_file():
        raise SystemExit(f"Prepared dataset not found: {dataset_root}")

    plan = get_lerobot_training_data_plan("panda_drawer")
    tag_metadata = _normalize_registered_lerobot_tag_metadata(TAG_METADATA_BY_TAG)
    stats_by_tag, repo_to_tag, _ = _collect_tagged_stats(
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

    print("Loading MolmoAct2 configuration and processor assets...")
    model_cfg = get_hf_model_config(
        "allenai/MolmoAct2",
        frame_loading_backend="av",
    )
    if not isinstance(model_cfg, MolmoAct2Config):
        raise TypeError(f"Expected MolmoAct2Config, got {type(model_cfg).__name__}")

    model_cfg.mm_preprocessor.max_subtitle_tokens = None
    model_cfg.mm_preprocessor.use_frame_special_tokens = True
    model_cfg.mm_preprocessor.max_frames = 8
    model_cfg.mm_preprocessor.max_fps = [2]
    model_cfg.mm_preprocessor.image.max_multi_image_crops = 8
    model_cfg.mm_preprocessor.image.max_images = 5
    model_cfg.max_action_dim = 8
    model_cfg.action_expert.max_action_dim = 8
    model_cfg.action_horizon = 30
    model_cfg.action_expert.max_horizon = 30
    model_cfg.n_obs_steps = 1
    model_cfg.add_action_expert = True
    model_cfg.action_format = "continuous"
    model_cfg.state_format = "continuous"
    model_cfg.mask_action_chunk_padding = True
    model_cfg.mask_action_dim_padding = True
    model_cfg.robot_processor = RobotProcessorConfig.from_stats(
        stats_by_tag=stats_by_tag,
        tag_metadata=tag_metadata,
        repo_to_tag=repo_to_tag,
        norm_mode="min_max",
    )

    assert model_cfg.add_action_expert
    assert model_cfg.max_action_dim == 8
    assert model_cfg.action_expert.max_action_dim == 8
    assert model_cfg.action_horizon == 30
    assert model_cfg.action_expert.max_horizon == 30
    assert model_cfg.n_obs_steps == 1
    assert model_cfg.robot_processor is not None

    print("Model type:", type(model_cfg).__name__)
    print("Action dimension:", model_cfg.max_action_dim)
    print("Action horizon:", model_cfg.action_horizon)
    print("Action expert enabled:", model_cfg.add_action_expert)

    wrapper = build_lerobot_dataset(
        f"lerobot:{DATASET_REPO_ID}",
        "train",
        n_obs_steps=1,
        max_action_horizon=30,
        max_action_dim=8,
    )
    preprocessor = model_cfg.build_preprocessor(
        for_inference=False,
        is_training=True,
        max_seq_len=4096,
        include_image=False,
    )
    processed_dataset = DeterministicDataset(
        dataset=wrapper,
        preprocessor=preprocessor,
        seed=6198,
        dataset_name=f"lerobot:{DATASET_REPO_ID}",
    )
    processed = processed_dataset[0]

    assert _shape(processed["states"]) == (8,)
    assert _shape(processed["actions"]) == (30, 8)
    assert np.isfinite(np.asarray(processed["states"])).all()
    assert np.isfinite(np.asarray(processed["actions"])).all()

    output_shapes = preprocessor.get_output_shapes()
    collator = model_cfg.build_collator(
        output_shapes,
        pad_mode="to_max",
        include_metadata=True,
    )
    batch = collator([processed])

    assert _shape(batch["states"]) == (1, 8)
    assert _shape(batch["actions"]) == (1, 30, 8)
    assert _shape(batch["action_horizon_is_pad"]) == (1, 30)
    assert _shape(batch["action_dim_is_pad"]) == (1, 8)

    print("\nProcessed example shapes:")
    for key in sorted(processed):
        shape = _shape(processed[key])
        if shape is not None:
            print(f"  {key}: {shape}")

    print("\nCollated batch shapes:")
    for key in sorted(batch):
        shape = _shape(batch[key])
        if shape is not None:
            print(f"  {key}: {shape}")

    print("\nPANDA MODEL CONFIGURATION AND COLLATION PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
