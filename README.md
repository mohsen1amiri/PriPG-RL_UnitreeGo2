







# PriPG-RL for Unitree Go2

> **Priority-guided Policy Gradient Reinforcement Learning for Autonomous Navigation of the Unitree Go2 Quadruped Robot**

https://private-user-images.githubusercontent.com/148786465/573299756-5a33b944-228e-4857-9487-6d0ff60dda69.mov

---

## Overview

This repository contains the implementation of **PriPG-RL**, a priority-guided reinforcement learning framework for training the [Unitree Go2](https://www.unitree.com/go2/) quadruped robot to perform autonomous point-to-point navigation. Training and evaluation are conducted in **NVIDIA Isaac Lab** (GPU-accelerated physics simulation), with a hierarchical policy design that decouples high-level navigation from low-level locomotion control.

The project benchmarks PriPG-RL (referred to as **P2P-SAC** in experiments) against three baselines — **SAC**, **PPO**, and **Accel. SAC** — across episodic reward and task success rate over 5M environment steps.

---

## Key Features

- **Point-to-point navigation** task built on the Isaac Lab manager-based environment framework
- **Hierarchical policy**: a pre-trained locomotion policy drives low-level joint control; a high-level RL agent commands navigation goals
- **Priority-guided reward shaping** to accelerate convergence and improve safety clearance
- **Multi-algorithm comparison**: P2P-SAC, Accel. SAC, SAC, PPO
- **SLURM-ready**: `sbatch` scripts included for HPC cluster training
- Reproducible seeds and structured CSV logging for training curves

---

## Repository Structure
```
PriPG-RL_UnitreeGo2/
├── IsaacLab/                          # Isaac Lab task definitions and environment configs
│   └── source/isaaclab_tasks/...      #   Navigation environments (P2P-SAC, SAC, PPO, Accel.)
├── locomotion_policy/                 # Pre-trained low-level locomotion policy for Go2
├── Codes_old/                         # Legacy/experimental code
├── sbatch_p2psac_shaped_*/            # SLURM job scripts — P2P-SAC with shaped reward
├── sbatch_shaped_grid_*/              # SLURM job scripts — shaped reward grid search
├── sbatch_frozenlake_cost_grid_*/     # SLURM job scripts — cost-grid ablation
└── README.md
```

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.10 |
| NVIDIA Isaac Lab | ≥ 2.0 |
| Isaac Sim | ≥ 4.2 |
| PyTorch | ≥ 2.2 (CUDA) |
| Stable-Baselines3 | ≥ 2.x |
| numpy, pandas | latest |

> A CUDA-capable GPU with ≥ 8 GB VRAM is required for simulation.

---

## Installation

**1. Install Isaac Lab** following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

**2. Clone this repository** into your Isaac Lab workspace:
```bash
git clone https://github.com/mohsen1amiri/PriPG-RL_UnitreeGo2.git
cd PriPG-RL_UnitreeGo2
```

**3. Register the custom task package:**
```bash
pip install -e IsaacLab/source/isaaclab_tasks
```

**4. Verify the environment is available:**
```bash
python -c "import isaaclab_tasks; print('Tasks registered successfully')"
```

---

## Training

### Local training (single run)
```bash
python IsaacLab/scripts/train.py \
    --task Isaac-Navigation-P2PSAC-Go2-v0 \
    --seed 0 \
    --num_envs 64
```

Replace `--task` with one of:
- `Isaac-Navigation-P2PSAC-Go2-v0` — P2P-SAC (proposed)
- `Isaac-Navigation-AccelSAC-Go2-v0` — Accel. SAC baseline
- `Isaac-Navigation-SAC-Go2-v0` — SAC baseline
- `Isaac-Navigation-PPO-Go2-v0` — PPO baseline

### HPC cluster (SLURM)

Pre-configured `sbatch` scripts are provided for each algorithm:
```bash
sbatch sbatch_p2psac_shaped_20260117_010250/run.sh
```

---

## Evaluation
```bash
python IsaacLab/scripts/eval.py \
    --task Isaac-Navigation-P2PSAC-Go2-v0 \
    --checkpoint <path/to/checkpoint.pt> \
    --num_envs 16
```

---

## Results

Training curves (mean ± 1 std over 2 seeds) comparing all algorithms over 5M environment steps:

| Algorithm | Final Reward (mean) | Final Success Rate |
|---|---|---|
| **P2P-SAC (ours)** | **~−400** | **~100%** |
| Accel. SAC | ~−600 | ~88% |
| SAC | ~−1130 | ~0% |
| PPO | ~−1140 | ~0% |

---

## Locomotion Policy

The `locomotion_policy/` directory contains a pre-trained low-level policy that maps velocity commands to joint torques for the Go2. This policy is kept frozen during high-level navigation training and was trained separately using standard locomotion reward formulations.

---

## License

This project is licensed under the [MIT License](LICENSE).


---

## Acknowledgements

- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) — simulation framework
- [Unitree Robotics](https://www.unitree.com/) — Go2 robot platform
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — RL algorithm implementations

