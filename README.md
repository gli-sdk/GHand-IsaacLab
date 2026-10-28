# GHand Isaac Lab

**English** | [简体中文](README.zh-CN.md)

Dexterous manipulation with the GHand5 right hand in Isaac Lab: cube reorientation, continuous cube rotation, and dual-ball rotation. Policies are trained with RL-Games PPO. Cube spin and dual-ball rotation also include joint-history student policies and EtherCAT deployment scripts.

## Tasks and demonstrations

| Cube Reorientation | Cube Spin | Dual-Ball Rotation |
|---|---|---|
| ![Cube reorientation](media/cube_reorientation.gif) | ![Cube spin](media/cube_spin.gif) | ![Dual-ball rotation](media/dual_ball_rotation.gif) |
| Random target orientations | Continuous rotation about an axis | Two balls rotating around each other |
| 36 mm cube, 26.45 g | Randomized 35–40 mm cubes, 30–50 g | 30 mm balls, 15 g each; horizontal hand |
| PPO · checkpoint iteration 12,000 | PPO + domain randomization · iteration 206 | PPO · iteration 200 |
| [MP4](media/cube_reorientation.mp4) | [MP4](media/cube_spin.mp4) | [MP4](media/dual_ball_rotation.mp4) |

Demos show approximately 15 seconds of simulation at normal playback speed. Iteration numbers are the counters stored in the displayed checkpoints, not comparable training budgets.

| Task directory | Isaac Lab task ID | Included models |
|---|---|---|
| `cube_reorientation` | `Isaac-Reorient-Cube-GHand-v0` | Teacher |
| `cube_spin` | `Isaac-Spin-Cube-GHand-v0` | Teacher and student |
| `dual_ball_rotation` | `Isaac-Dual-Ball-Rotation-GHand-v0` | Teacher and student |

Each task has its own environment, rewards, observations and policy weights. The hand asset and common utilities are shared. Policies are not interchangeable between tasks.

## Setup

Validated with **Isaac Lab 2.3.0 / Isaac Sim 5.1.0**. Install Isaac Lab separately, then run commands from this repository's root:

```bash
export ISAACLAB_PATH="$HOME/workspace/IsaacLab"
# Optional editable installation:
bash python.sh -m pip install --no-deps --no-build-isolation -e .
```

`python.sh` adds `source/` and the Isaac Lab packages to the Python path. It defaults to `$ISAACLAB_PATH/_isaac_sim/python.sh`; use `ISAACSIM_PYTHON` to override the launcher. Keep `assets/`, `config/` and `models/` alongside `source/`: a Python wheel alone does not contain the complete runtime resources.

## Simulation

```bash
# Cube reorientation, including the reference target
bash scripts/cube_reorientation/play.sh 4

# Continuous cube rotation
bash python.sh scripts/cube_spin/evaluate.py \
  --checkpoint models/cube_spin/teacher/nn/policy.pth \
  --num_envs 4 --episodes_per_env 2 --output runs/cube_spin/view_teacher.json

# Dual-ball rotation
bash python.sh scripts/dual_ball_rotation/evaluate.py \
  --checkpoint models/dual_ball_rotation/teacher/nn/policy.pth \
  --num_envs 4 --episodes 3
```

Student simulation:

```bash
bash python.sh scripts/cube_spin/distill.py \
  --student models/cube_spin/student/student.pt \
  --num_envs 4 --episodes-per-env 1 --eval-output runs/cube_spin/view_student.json

bash python.sh scripts/dual_ball_rotation/evaluate.py \
  --student models/dual_ball_rotation/student/student.pt \
  --num_envs 4 --episodes 3
```

Recording is off by default. Add `--headless` to run without a window. Cube and ground resources are loaded from `assets/official/`.

## Training and distillation

```bash
bash python.sh scripts/cube_reorientation/train.py --headless --num_envs 4096
bash python.sh scripts/cube_spin/train.py --headless --num_envs 4096
bash python.sh scripts/dual_ball_rotation/train.py --headless --num_envs 1024 --max_iterations 300 --seed 42

bash python.sh scripts/cube_spin/distill.py --headless \
  --teacher models/cube_spin/teacher/nn/policy.pth \
  --run-dir runs/cube_spin/student_new --num_envs 256 --iterations 300

bash python.sh scripts/dual_ball_rotation/distill.py --headless \
  --teacher models/dual_ball_rotation/teacher/nn/policy.pth \
  --run-dir runs/dual_ball_rotation/student_new --num_envs 256 --iterations 300
```

PPO settings live in `config/<task>/ppo.yaml`; environment settings are defined by each task's `env_cfg.py`. Cube tasks additionally load `config/<task>/env.yaml`. Fresh cube-spin training uses fixed-object defaults; the included randomized model's size, mass and friction ranges are recorded in `models/cube_spin/teacher/params/env.yaml`. For dual-ball training, `GHAND_DUAL_BALL_BASE_CONFIG` optionally selects a saved environment YAML.

Evaluation restores the selected model's `params/env.yaml` before applying explicit CLI overrides. The included `params/agent.yaml` records its PPO settings. These saved configurations and deployment contracts retain historical training metadata. When changing environment counts, satisfy PPO minibatch divisibility requirements.

The vision-free students learn from privileged teachers using joint observations and command history. They do not receive object pose from a camera. The retained dual-ball student achieved 63/64 successful simulation episodes with seed 123 and mean 7.86 revolutions; success requires at least two revolutions within 30 seconds without dropping a ball. See [evaluation](models/dual_ball_rotation/student/evaluation.json). This is a fixed-initial-condition simulation result, not a measured hardware success rate.

## Hardware deployment

Use a Python environment with PyTorch, PyYAML, the GHand Python SDK and EtherCAT support. Device motion requires `--execute`. Run the preparation steps with an empty, secured hand and an on-site operator.

```bash
export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
export GHAND_SDK_ROOT="$HOME/ghand/GHand-Python-SDK"
export GHAND_ETHERCAT_INTERFACE=enx00e04c665f71  # Replace with your EtherCAT interface

# Cube spin: offline policy check, then empty-hand reset
python scripts/cube_spin/deploy.py \
  --policy-dir models/cube_spin/student \
  --output "runs/cube_spin/offline_$(date +%Y%m%d_%H%M%S).json"
python scripts/cube_spin/reset_hand.py --hand-side right --execute

# Place the cube and withdraw your hands before executing
python scripts/cube_spin/deploy.py --hand-side right \
  --policy-dir models/cube_spin/student \
  --execute --object-present --duration 10 --speed 15 --torque 20 --onsite-cutoff-ready \
  --output "runs/cube_spin/hardware_$(date +%Y%m%d_%H%M%S).json"
```

```bash
# Dual-ball rotation: offline policy check, then saved empty-hand start pose
python scripts/dual_ball_rotation/deploy.py --policy-dir models/dual_ball_rotation/student
python scripts/dual_ball_rotation/prepare_hand.py --policy-dir models/dual_ball_rotation/student \
  --execute --empty-hand --speed 25 --torque 25 --onsite-cutoff-ready

# Secure the hand horizontally, place both balls and withdraw your hands
python scripts/dual_ball_rotation/deploy.py --policy-dir models/dual_ball_rotation/student \
  --execute --object-present --duration 10 --speed 25 --torque 25 --onsite-cutoff-ready
```

Both students issue position commands at 20 Hz. SDK speed and torque percentages have not been calibrated to simulation physical units. Observe the objects externally to assess task success. Cube reorientation has no hardware deployment entry point. Dual-ball deployment overwrites `runs/dual_ball_rotation/hardware/latest.json`; pose preparation has a separate log.

## Code structure

```text
assets/                        # One hand asset, joint manifest, local cube/ground assets
config/<task>/                 # PPO, environment and hardware settings
source/ghand/common/           # Hand loading, config restoration, joint and cube utilities
source/ghand/tasks/<task>/     # Task environment, rewards, student and deployment adapters
scripts/<task>/                # Train, evaluate, distill, deploy and prepare/reset entry points
models/<task>/teacher/         # PPO weights and matching configuration
models/<task>/student/         # Student weights, TorchScript actor and deployment contract
media/                         # Simulation GIFs and MP4s
tests/<task>/                  # Core offline regression tests
```

In each task, `env.py` implements simulation behavior and `env_cfg.py` defines configuration. Cube spin adds `core.py`, `randomization.py` and `student.py`. Dual-ball rotation uses `orbit.py` for rotation metrics, `palm_frame.py` for palm coordinates, `perturbations.py` for randomization, `handover.py` for distillation scheduling, and `deployment.py` / `start_pose.py` for policy execution and pose preparation. Shared cube physics and mimic-joint resets are in `common/inhand_env.py`, `cube_env.py` and `cube_env_cfg.py`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=source python -m unittest discover -s tests
```

Training logs, evaluation results and caches are excluded by `.gitignore`. Model weights and demonstration videos are retained.

## License

Source code is licensed under [BSD-3-Clause](LICENSE). GHand USD and NVIDIA resources belong to their respective providers and are not covered by this source code license.
