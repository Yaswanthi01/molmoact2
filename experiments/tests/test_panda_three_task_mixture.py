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
