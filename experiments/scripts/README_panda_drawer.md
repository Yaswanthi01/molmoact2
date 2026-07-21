# Panda Drawer MolmoAct2 Pipeline

This branch adds a local pipeline for fine-tuning MolmoAct2 on the LeRobot v3.0 Panda drawer dataset:

```text
kitalr/panda_drawer_many_ee_fullres_v3.0
```

The dataset itself is not committed to git. Each user should download it locally or on the training machine and run the preparation script.

## Quick Start

These steps assume the repository has already been cloned.

From the repository root, create and activate a Python 3.12 environment:

```bash
cd /path/to/molmoact2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the project requirements into that environment:

```bash
cd experiments
pip install -e ".[all]"
pip install -e "./lerobot[async,libero]"
pip install debugpy pandas pyarrow av
```

If your machine does not have `python3.12`, install Python 3.12 first. LeRobot in this repo expects Python `>=3.12`.

Then download, prepare, and validate the dataset:

```bash
cd /path/to/molmoact2
mkdir -p datasets

huggingface-cli download kitalr/panda_drawer_many_ee_fullres_v3.0 \
  --repo-type dataset \
  --local-dir datasets/panda_drawer_many_ee_fullres_v3.0

python experiments/scripts/prepare_panda_drawer_dataset.py \
  --source datasets/panda_drawer_many_ee_fullres_v3.0 \
  --output datasets/panda_drawer_ee_molmoact2_v3 \
  --overwrite

python experiments/scripts/validate_panda_videos.py
python experiments/scripts/preflight_panda_wrapper.py
python experiments/scripts/preflight_panda_model.py
```

For a local machine where disk space matters, you can symlink videos instead of copying them:

```bash
python experiments/scripts/prepare_panda_drawer_dataset.py \
  --source datasets/panda_drawer_many_ee_fullres_v3.0 \
  --output datasets/panda_drawer_ee_molmoact2_v3 \
  --overwrite \
  --symlink-videos
```

Use real copied videos on clusters unless you are sure the symlink target will exist on the compute node.

Expected success messages:

```text
FULL VIDEO VALIDATION PASSED
MOLMOACT2 WRAPPER PREFLIGHT PASSED
PANDA MODEL CONFIGURATION AND COLLATION PREFLIGHT PASSED
```

The virtual environment is intentionally not committed. Everyone using this repo should create their own environment and install the requirements with the commands above.

If the dataset is private or gated, log in first:

```bash
huggingface-cli login
```

## Local Smoke Test

On a Mac or CPU-only machine, the actual trainer is expected to fail at CUDA. This is still useful because it proves the command reaches the GPU boundary.

From `experiments/`:

```bash
cd experiments
source ../.venv/bin/activate

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export GLOO_SOCKET_IFNAME=lo0
export LEROBOT_DATA_ROOT="../datasets"
export LEROBOT_VIDEO_BACKEND=pyav

torchrun \
  --nnodes=1 \
  --nproc-per-node=1 \
  --master-addr=127.0.0.1 \
  --master-port=29501 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  panda_drawer_ee \
  --max_duration=1 \
  --log_interval=1 \
  --device_batch_size=1 \
  --global_batch_size=1 \
  --num_workers=0 \
  --pin_memory=false \
  --save_folder="../checkpoints/panda-smoke" \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none
```

Expected local CPU/Mac endpoint:

```text
torch.cuda.set_device(...)
AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'
```

That CUDA error is expected locally. Real fine-tuning requires an NVIDIA CUDA GPU.

## SLURM Smoke Test

After setting up the repo, environment, and prepared dataset on a GPU cluster:

```bash
cd ~/molmoact2/experiments
sbatch slurm/panda_smoke.slurm
```

Check status:

```bash
squeue -u "$USER"
```

Check logs:

```bash
tail -f logs/molmoact2-panda-smoke-*.out
```

Edit `experiments/slurm/panda_smoke.slurm` if your cluster uses different account, partition, memory, or path names.

## Added Files

```text
experiments/launch_scripts/data_mixtures.py
```

Registers the `panda_drawer_ee` MolmoAct2/LeRobot training mixture. It maps the prepared dataset to:

- state key: `observation.state`
- action key: `action`
- cameras:
  - `observation.images.gripper_cam.rgb`
  - `observation.images.left_cam.left`
  - `observation.images.right_cam.left`
- action dimension: `8`
- action horizon: `30`
- control mode: absolute end-effector pose

```text
experiments/scripts/prepare_panda_drawer_dataset.py
```

Creates the MolmoAct2-ready local dataset:

```text
datasets/panda_drawer_ee_molmoact2_v3
```

from the source dataset:

```text
datasets/panda_drawer_many_ee_fullres_v3.0
```

It leaves the source dataset untouched. It adds:

```text
observation.state = observation.proprio.joint_pos + observation.proprio.gripper
action = action.ee_pos + action.ee_rot + action.gripper
```

Both output vectors have shape `(8,)`. State contains 7 Panda joint positions plus
1 gripper value; action contains 3 end-effector positions, the source's 4D rotation representation, and
1 gripper value. `action.joint_pos` and all `observation.target.*` fields are ignored.

```text
experiments/scripts/validate_panda_videos.py
```

Decodes every MP4 in the prepared dataset and checks:

- all camera directories exist
- every video decodes
- resolution matches metadata
- FPS matches metadata
- codec metadata matches actual AV1 decoding
- each camera has `71019` decoded frames

```text
experiments/scripts/preflight_panda_wrapper.py
```

Builds the real MolmoAct2 LeRobot wrapper and checks:

- wrapper can load `panda_drawer_ee_molmoact2_v3`
- state shape is `(8,)`
- action chunk shape is `(30, 8)`
- three images are returned
- episode-end action padding behaves correctly

```text
experiments/scripts/preflight_panda_model.py
```

Loads the MolmoAct2 config/processor path and checks:

- MolmoAct2 action dimension is `8`
- action horizon is `30`
- action expert is enabled
- one Panda example can be preprocessed
- collator builds a valid batch

```text
experiments/scripts/serve_policy.py
```

Adds the `panda_drawer_ee` camera preset for inference/server use.

```text
experiments/slurm/panda_smoke.slurm
```

Example SLURM smoke-test job for one GPU. Adjust account, partition, memory, and paths if your cluster differs.

## Notes

- Do not commit `datasets/`, `checkpoints/`, `logs/`, `.cache/`, or virtual environments.
- The prepared dataset is local and reproducible from the Hugging Face source dataset.
- The training mixture name is `panda_drawer_ee`.
- The prepared LeRobot dataset id is `panda_drawer_ee_molmoact2_v3`.
- The model checkpoint used in the smoke commands is `allenai/MolmoAct2`.
