import pytest

from launch_scripts.data_mixtures import TAG_METADATA_BY_TAG
from launch_scripts.train_lerobot import get_lerobot_training_data_plan


def test_panda_three_task_absolute_uses_only_requested_full_datasets():
    plan = get_lerobot_training_data_plan("panda_three_task_absolute")

    assert len(plan.robot_mixture) == 1
    group = plan.robot_mixture[0]
    assert group.name == "lerobot:panda_ee_state_absolute"
    assert [dataset.dataset_name for dataset in group.datasets] == [
        "lerobot:panda_drawer_ee_state_absolute_molmoact2_v3",
        "lerobot:panda_sort_three_continued_ee_fullres_full_v30",
        "lerobot:panda_stack_two_cup_ee_state_absolute_molmoact2_v3",
    ]

    metadata = TAG_METADATA_BY_TAG[group.name]
    assert metadata["action_dim"] == 8
    assert metadata["action_horizon"] == 30


def _dataset_rates(name):
    plan = get_lerobot_training_data_plan(name)
    return {
        group.datasets[0].dataset_name: float(group.rate)
        for group in plan.robot_mixture
    }


def test_panda_three_task_window_proportional_uses_valid_window_counts():
    rates = _dataset_rates("panda_three_task_window_proportional")
    total_windows = 153_420 + 51_862 + 68_090

    assert rates["lerobot:panda_sort_three_continued_ee_fullres_full_v30"] == pytest.approx(
        153_420 / total_windows
    )
    assert rates["lerobot:panda_stack_two_cup_ee_state_absolute_molmoact2_v3"] == pytest.approx(
        51_862 / total_windows
    )
    assert rates["lerobot:panda_drawer_ee_state_absolute_molmoact2_v3"] == pytest.approx(
        68_090 / total_windows
    )
    assert sum(rates.values()) == pytest.approx(1.0)


def test_panda_three_task_equal_gives_each_task_equal_priority():
    rates = _dataset_rates("panda_three_task_equal")

    assert all(rate == pytest.approx(1.0 / 3.0) for rate in rates.values())
    assert sum(rates.values()) == pytest.approx(1.0)
