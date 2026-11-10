# Waypoint-Based RL with Self-Supervised Dense Reward Models

Teaching a 7-DoF Franka Panda arm to solve multi-step manipulation **without a hand-engineered reward function**. A self-supervised ensemble reward model (SORS) scores partial progress, and waypoints are optimized against it with SLSQP.

Built on [robosuite](https://robosuite.ai/) / MuJoCo with PyTorch.

---

## Demo

**Stack** — place cube A on cube B

| Baseline (sparse reward) | Ours (SORS waypoints) |
|---|---|
| ![Stack baseline](assets/Stack_org.gif) | ![Stack ours](assets/Stack_sors.gif) |

**Lift** — grasp and lift the cube

| Baseline (sparse reward) | Ours (SORS waypoints) |
|---|---|
| ![Lift baseline](assets/Lift_org.gif) | ![Lift ours](assets/Lift_sors.gif) |

---

## The problem

Sparse-reward manipulation is hard to learn: the agent is only rewarded on success, which almost never happens by chance in a multi-step task like stacking. The usual workaround is hand-engineering a dense reward, but that takes per-task domain knowledge and frequently rewards the wrong behaviour.

This project replaces the hand-engineered reward with a reward model the agent learns **from its own experience**, then plans waypoints against it.

## Approach

### Self-supervised reward model (SORS)

An ensemble of **10 `RNetwork` MLPs** (hidden dim 128) is trained on trajectories from a global replay buffer, each labelled with its sparse episode return. The ensemble mean becomes the dense reward signal; disagreement between members suppresses the outlier scores that are the main source of noise in self-supervised reward learning.

Training runs on a periodic schedule: every `Pr` episodes, `Nr` minibatch updates are applied across the ensemble.

### Three-phase waypoint scheduling

Each waypoint is learned in phases, driven by a global episode counter:

| Phase | Behaviour |
|---|---|
| `exploration_epoch` | Purely stochastic exploration of the current waypoint |
| `ensemble_sampling_epoch` | A random ensemble member is sampled and used as the optimization target |
| after that | Optimization against the **ensemble mean** |
| `averaging_noise_epoch` | Averaging / noise phase to keep exploring around converged solutions |

Waypoints are optimized with **SLSQP** (`scipy.optimize.minimize` under linear constraints), initialized from the best trajectory buffer and from the ensembles learned for previous waypoints.

### Setup

| | |
|---|---|
| Robot | Franka Panda |
| Controller | `OSC_POSE`, 10 Hz |
| Reward | Sparse task reward (`reward_shaping=False`) + small shaped grasp bonus |
| Tasks | `Lift` (state_dim 4, wp_steps 50, gripper_steps 10) · `Stack` (state_dim 4, wp_steps 50, gripper_steps 15) |

## Results

Compared against four baselines — **PPO-vanilla, PPO-waypoints, SAC-vanilla, SAC-waypoints**:

| Task | Metric | Baseline | Ours |
|---|---|---|---|
| Lift | Success rate | `[fill in]` | `[fill in]` |
| Lift | Episodes to converge | `[fill in]` | `[fill in]` |
| Stack | Success rate | `[fill in]` | `[fill in]` |
| Stack | Episodes to converge | `[fill in]` | `[fill in]` |

<!-- Điền từ eval_data.pkl trong rlwp/models/*/test/, hoặc từ plot_rewards.py.
     Nếu có learning curve, thêm: ![learning curves](assets/curves.png)
     Xóa dòng comment này trước khi commit. -->

## Running it

```bash
cd rlwp

# train
python main.py train=True task=Lift  run_name=sors_lift_<seed>
python main.py train=True task=Stack run_name=sors_stack_<seed>

# evaluate a trained model
python main.py test=True task=<task_name> run_name=test render=True
```

Full argument set lives in `rlwp/cfg/`. Pretrained models for each task are included under `rlwp/models/`.

## Repository layout

```
├── rlwp/                  # main implementation
│   ├── main.py            # entry point
│   ├── method.py          # SORS ensemble + SLSQP waypoint optimization
│   ├── memory.py          # global trajectory replay buffer
│   ├── models/            # pretrained models per task
│   ├── cfg/               # task + run configuration
│   └── README.md          # full reproducibility notes
├── Baselines/             # PPO / SAC, vanilla and waypoint variants
├── robosuite/             # vendored simulator
└── assets/                # demo GIFs
```

## Installation

Developed on Windows 11 with Anaconda. Full dependency list in `requirements.txt`.

```bash
conda create -n robosuite python=3.8
conda activate robosuite
pip install robosuite robomimic
pip install -r requirements.txt
```

MuJoCo 2.1 must be installed locally — extract the `mujoco210` directory into `~/.mujoco/mujoco210`.

<details>
<summary>Windows-specific notes</summary>

- robosuite installation guide: https://robosuite.ai/docs/installation.html
- If `egl` fails to build, install the patched build:
  `pip install https://github.com/mhandb/egl_probe/archive/fix_windows_build.zip`
  (see https://github.com/huggingface/lerobot/issues/105)
- Visual Studio Professional 2019 may be required for MuJoCo 2.1 compatibility.

</details>

---

Course project, COMP579 — Reinforcement Learning, McGill University.
