from __future__ import annotations


def _entry(task: str, mode: str, runs: tuple[str, ...], checkpoint: str, step: int) -> dict:
    return {
        "task": task,
        "mode": mode,
        "runs": runs,
        "checkpoint": checkpoint,
        "step": step,
        "output": f"{task}/{mode}/step{step}",
    }


CHECKPOINTS: list[dict] = []

for step in (10000, 20000, 30000, 50000, 60000, 70000, 80000, 90000):
    CHECKPOINTS.append(
        _entry(
            "drawer",
            "full",
            (
                "panda-drawer-ee-state-absolute-full-h30s30-bs32-2gpu-120k",
                "panda-drawer-ee-state-absolute-full-h30s30-bs32-120k",
            ),
            f"step{step}",
            step,
        )
    )
    CHECKPOINTS.append(
        _entry(
            "cup",
            "full",
            (
                "panda-stack-two-cup-ee-state-absolute-full-h30s30-bs32-2gpu-120k",
                "panda-stack-two-cup-ee-state-absolute-full-h30s30-bs32-120k",
            ),
            f"step{step}",
            step,
        )
    )

for step in (10000, 20000, 30000, 50000, 60000, 70000, 80000):
    CHECKPOINTS.append(
        _entry(
            "drawer",
            "lora",
            ("panda-drawer-ee-state-absolute-lora-h30s30-bs32-120k",),
            f"step{step}-merged",
            step,
        )
    )
    CHECKPOINTS.append(
        _entry(
            "cup",
            "lora",
            ("panda-stack-two-cup-ee-state-absolute-lora-h30s30-bs32-120k",),
            f"step{step}-merged",
            step,
        )
    )

for step in (10000, 20000, 30000, 40000, 50000, 60000):
    CHECKPOINTS.append(
        _entry(
            "sort",
            "full",
            (
                "panda-sort-three-absolute-all-full-h30s30-bs64-4gpu-120k",
                "panda-sort-three-absolute-all-full-h30s30-bs64-120k",
            ),
            f"step{step}",
            step,
        )
    )

for step in (10000, 20000, 30000, 40000):
    CHECKPOINTS.append(
        _entry(
            "sort",
            "lora",
            ("panda-sort-three-absolute-all-lora-action-h30s30-bs64-120k",),
            f"step{step}-merged",
            step,
        )
    )

# Additional checkpoints selected after the original 40-checkpoint conversion.
for step in (30000, 40000, 50000):
    CHECKPOINTS.append(
        _entry(
            "multitask",
            "full",
            ("panda-multitask-absolute-full-h30s30-bs64-4gpu-120k",),
            f"step{step}",
            step,
        )
    )

CHECKPOINTS.append(
    _entry(
        "multitask",
        "lora",
        ("panda-multitask-absolute-lora-action-h30s30-bs64-120k",),
        "step40000-merged",
        40000,
    )
)

CHECKPOINTS.append(
    _entry(
        "cup",
        "full",
        ("panda-stack-two-cup-ee-state-absolute-full-h30s30-bs32-2gpu-120k",),
        "step40000",
        40000,
    )
)
CHECKPOINTS.append(
    _entry(
        "cup",
        "full",
        ("panda-stack-two-cup-ee-state-absolute-full-h30s30-bs32-2gpu-120k",),
        "step100000",
        100000,
    )
)
CHECKPOINTS.append(
    _entry(
        "cup",
        "lora",
        ("panda-stack-two-cup-ee-state-absolute-lora-h30s30-bs32-120k",),
        "step40000-merged",
        40000,
    )
)
CHECKPOINTS.append(
    _entry(
        "drawer",
        "full",
        ("panda-drawer-ee-state-absolute-full-h30s30-bs32-2gpu-120k",),
        "step40000",
        40000,
    )
)
CHECKPOINTS.append(
    _entry(
        "drawer",
        "lora",
        ("panda-drawer-ee-state-absolute-lora-h30s30-bs32-120k",),
        "step40000-merged",
        40000,
    )
)

# Three-task sampling-weight ablation checkpoints.
for step in (10000, 20000):
    CHECKPOINTS.append(
        _entry(
            "three-task",
            "alpha0-equal",
            ("panda-three-task-equal-full-h30s30-bs64-4gpu-120k",),
            f"step{step}",
            step,
        )
    )
    CHECKPOINTS.append(
        _entry(
            "three-task",
            "alpha05-absolute",
            ("panda-three-task-absolute-full-h30s30-bs64-4gpu-120k",),
            f"step{step}",
            step,
        )
    )
    CHECKPOINTS.append(
        _entry(
            "three-task",
            "alpha1-window-proportional",
            ("panda-three-task-window-proportional-full-h30s30-bs64-4gpu-120k",),
            f"step{step}",
            step,
        )
    )

assert len(CHECKPOINTS) == 55


def selected(start: int, end: int) -> list[tuple[int, dict]]:
    if not 0 <= start <= end < len(CHECKPOINTS):
        raise ValueError(f"expected 0 <= start <= end <= {len(CHECKPOINTS) - 1}")
    return list(enumerate(CHECKPOINTS[start : end + 1], start=start))


def source_candidates(entry: dict) -> list[str]:
    return [
        f"{run}{suffix}/{entry['checkpoint']}"
        for run in entry["runs"]
        for suffix in ("-milestones", "")
    ]
