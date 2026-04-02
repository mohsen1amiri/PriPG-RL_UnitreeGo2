# PriPG-RL: Privileged Planner-Guided Reinforcement Learning for the Unitree Go2

> **Privileged Planner-Guided Reinforcement Learning for Partially Observable Systems with Anytime Feasible MPC**
>
> Mohsen Amiri¹‡, Mohsen Amiri²‡, Ali Beikmohammadi¹, Sindri Magnússon¹, and Mehdi Hosseinzadeh², *Senior Member, IEEE*
>
> ¹ Department of Computer and System Science, Stockholm University, Sweden  
> ² School of Mechanical and Material Engineering, Washington State University, USA  
> ‡ Equal contribution

https://private-user-images.githubusercontent.com/148786465/573299756-5a33b944-228e-4857-9487-6d0ff60dda69.mov

---

## Overview

This repository contains the official implementation of **PriPG-RL** (Privileged Planner-Guided Reinforcement Learning), a framework for training reactive RL policies under **partial observability** by exploiting a privileged, anytime-feasible planner agent that is available exclusively during training.

The core problem is formalized as a **Partially Observable Markov Decision Process (POMDP)**, where a planner agent with access to an approximate dynamical model and privileged state information guides a learning agent that observes only a lossy projection of the true state. Standard RL methods (SAC, PPO, TD3) frequently fail in this setting due to state aliasing, which causes uninformative early exploration, critic instability, and convergence to suboptimal local minima.

PriPG-RL consists of two instantiations:
- **REAP-based Planner Agent** — a feasibility-preserving approximate MPC that provides always-feasible guidance at controllable computational cost via a primal–dual gradient flow.
- **P2P-SAC** (Planner-to-Policy Soft Actor-Critic) — the learning agent that distills privileged planner knowledge through four mechanisms: a dual replay buffer, a three-phase maturity schedule, a logit-space imitation anchor, and an advantage-based sigmoid gate.

The framework is validated in **NVIDIA Isaac Lab** simulation and successfully deployed on a real-world **Unitree Go2** quadruped navigating complex, obstacle-rich environments.

---

## Key Contributions

- **PriPG-RL framework**: a general formulation for planner-guided actor–critic RL under partial observability with rigorous theoretical analysis, including a proof that the P2P-SAC actor gradient separates useful privileged guidance from irreducible aliasing variance (Theorem 1).
- **REAP-based planner agent**: an anytime-feasible MPC that guarantees constraint satisfaction at any computation budget via a modified barrier function and primal–dual gradient flow, serving as a structured, online training signal.
- **P2P-SAC algorithm**: four tightly coupled mechanisms that exploit the planner's guidance without bounding the asymptotic performance of the learned policy:
  - *Dual replay buffer*: a write-once planner buffer + standard FIFO online buffer for mixed mini-batch sampling.
  - *Three-phase maturity schedule*: plateau → annealing → maturity, with a non-zero final guidance coefficient to prevent catastrophic return to unmitigated state aliasing.
  - *Logit-space imitation anchor*: resolves vanishing gradients near the control boundary ∂𝔘 that plague output-space losses in prior work.
  - *Advantage-based sigmoid gate*: selectively suppresses imitation where the learned policy dominates and preserves it where the planner's privileged information confers an unreplicable advantage.
- **100% success rate** on a challenging POMDP navigation task (blind navigation with 6 cylindrical obstacles) where standard SAC and PPO fail entirely.
- **Zero-shot sim-to-real transfer** validated on physical Unitree Go2 hardware.

---

## Repository Structure

```
PriPG-RL_UnitreeGo2/
├── IsaacLab/                          # Isaac Lab task definitions and environment configs
│   └── source/isaaclab_tasks/         #   Navigation task (P2P-SAC, Accel. SAC, SAC, PPO)
├── locomotion_policy/                 # Frozen low-level locomotion policy (RSL-RL, 200 Hz)
│                                      #   Maps [vx, vy] velocity commands → joint torques
├── Codes_old/                         # Legacy and experimental code
├── sbatch_p2psac_shaped_*/            # SLURM scripts — P2P-SAC with shaped reward
├── sbatch_shaped_grid_*/              # SLURM scripts — shaped reward grid search
├── sbatch_frozenlake_cost_grid_*/     # SLURM scripts — cost-grid ablation (FrozenLake)
├── .gitignore
├── LICENSE                            # MIT License
└── README.md
```

---

## Method

### Problem Setting

The Unitree Go2 operates under a **POMDP** induced by a lossy observation map. The learning agent's observation is:

```
s_t = [x_rob, y_rob, x_goal, y_goal] ∈ R^4
```

Obstacle positions, heading, and joint quantities are **excluded** from `s_t` (blind navigation), inducing state aliasing. The planner agent additionally receives privileged information `I_t = {obstacle geometry, boundary geometry, goal}`, which is never communicated to the learning agent at deployment.

### REAP-Based Planner Agent

The planner operates on a 2D single-integrator approximation (`z_{k+1} = z_k + u_k · 0.02` at 50 Hz) with tightened obstacle-avoidance halfspace constraints. It solves a finite-horizon MPC problem (horizon N=15) by evolving a continuous-time primal–dual gradient flow:

```
dû/dρ = −ζ ∇_û B(z, d, û, λ̂)
dλ̂/dρ = +ζ (∇_λ̂ B(z, d, û, λ̂) + Ψ_ρ)
```

The modified barrier function ensures the solution trajectory remains **feasible at any termination time** ρ, with solution quality improving monotonically as more computation is allocated (`t_cpu`). This is formally validated by two propositions: exponential convergence to the optimal solution, and invariance of the feasible set.

### P2P-SAC

P2P-SAC augments the standard SAC actor objective with a gated logit-space anchor:

```
L_π(θ) = L_SAC(θ) + β_t · G_φ(s, u†; M_t) · ℓ(s, u†)
```

where:
- `ℓ(s, u†) = (1/p) ‖μ_θ(s) − ξ†‖²` is the logit-space imitation loss (gradient non-vanishing near ∂𝔘)
- `G_φ = (1 − M_t) + M_t · σ(Â†(s, u†) / τ_g)` is the composite advantage gate
- `β_t` follows the three-phase schedule, with `β_f > 0` maintained at maturity to prevent catastrophic forgetting

---

## Simulation Setup

| Parameter | Value |
|---|---|
| Simulator | NVIDIA Isaac Lab (`Isaac-Velocity-Flat-Unitree-Go2-v0`) |
| Arena | 4.1 × 5.6 m², x ∈ [−2.0, 2.0] m, y ∈ [−2.0, 3.5] m |
| Obstacles | 6 cylinders, radius 0.23 m (symmetric layout) |
| Control frequency | 50 Hz (high-level), 200 Hz (locomotion) |
| Action space | u_t = [v_x, v_y]ᵀ ∈ [−0.5, 0.5]² m/s |
| Episode termination | Goal (< 0.3 m), collision, fall (trunk < 0.1 m), timeout (8,000 steps) |
| Reward | r = −1.0 − 0.02‖u_t‖² + 100 (goal) − 200 (crash) |
| Hardware | NVIDIA A40 GPU |
| Seeds | 5 per algorithm {0, …, 4} |

---

## Results

### Training Curves (5M environment steps, mean ± 1 std)

P2P-SAC achieves **100% success after ~1M steps**. Accelerated SAC reaches ~40% at 5M steps. Vanilla SAC and PPO fail entirely, operating in the unmitigated POMDP with persistent state aliasing.

### Best-Checkpoint Evaluation (mean ± std, 5 seeds)

| Algorithm | Success (%) | Crash (%) | Path Optimality | Runtime (s) | Avg. Velocity (m/s) |
|---|---|---|---|---|---|
| SAC | 0.0 | — | — | — | — |
| PPO | 0.0 | 100.0 | — | — | — |
| Accel. SAC | 35.0 ± 47.7 | 65.0 ± 47.7 | 1.100 ± 0.073 | 9.0 ± 1.3 | 0.477 ± 0.045 |
| **P2P-SAC (ours)** | **100.0** | **0.0** | **1.060 ± 0.031** | **9.7 ± 1.1** | **0.352 ± 0.019** |
| REAP (planner only) | 100.0 | 0.0 | 1.10 ± 0.04 | 12.26 ± 1.29 | 0.353 ± 0.028 |

Path optimality is defined as `episode length / straight-line distance to goal` (1.0 = optimal straight path).

---

## Hardware Deployment

The policy is deployed zero-shot on a physical Unitree Go2 quadruped:

- **Remote compute**: Intel i9-13900K, 64 GB RAM (Wi-Fi link)
- **State estimation**: OptiTrack system (10 Prime×13 cameras, 120 Hz, ±0.02 mm accuracy)
- **Control loop**: 50 Hz closed-loop

The robot successfully avoids all obstacles and reaches the goal, demonstrating that the P2P-SAC policy transfers to hardware under real-world conditions.

---

## Installation

### Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.10 |
| NVIDIA Isaac Lab | ≥ 2.0 |
| Isaac Sim | ≥ 4.2 |
| PyTorch | ≥ 2.2 (CUDA) |
| Stable-Baselines3 | ≥ 2.x |
| numpy, pandas | latest |

> A CUDA-capable GPU with ≥ 8 GB VRAM is required for simulation.

### Setup

**1. Install Isaac Lab** following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

**2. Clone this repository:**
```bash
git clone https://github.com/mohsen1amiri/PriPG-RL_UnitreeGo2.git
cd PriPG-RL_UnitreeGo2
```

**3. Register the task package:**
```bash
pip install -e IsaacLab/source/isaaclab_tasks
```

---

## Training

All algorithms are launched via the `isaacrun` wrapper from inside the `IsaacLab/` root directory. First, set up the environment:

```bash
export ISAACLAB_ROOT="/path/to/IsaacLab"
export COMPASS_ROOT="${ISAACLAB_ROOT}/compass_algorithm"
cd "${ISAACLAB_ROOT}"
source "${COMPASS_ROOT}/isaac_helpers.sh"

export PORTABLE_ROOT="${COMPASS_ROOT}/kit_portable/run_seed0"
mkdir -p "${PORTABLE_ROOT}/kit_cache" "${PORTABLE_ROOT}/kit_data" \
         "${PORTABLE_ROOT}/kit_logs"  "${PORTABLE_ROOT}/kit_documents"
```

### P2P-SAC (proposed)

```bash
isaacrun compass_algorithm/train_compass_v4.py --headless \
    --device cuda:0 \
    --algo compass_v3 \
    --seed 0 \
    --num_envs 1 \
    --total_timesteps 50_000_000 \
    --video --video_length 2_000 --video_interval 100_000 \
    --cv2_beta_0 10.0 --cv2_beta_final 10.0 \
    --cv2_plateau_steps 100_000 --cv2_decay_steps 0 \
    --buffer_size 2_000_000 \
    --gate_tau 1.0 \
    --action_mode xy \
    --max_steps 8_000 \
    --goal_reward 100.0 --crash_cost 200.0 --step_cost 1.0 \
    --goal_tol 0.3 --robot_radius 0.3 \
    --leg_reach_margin 0.02 --wall_margin 0.35 \
    --vx_max 0.7 --vy_max 0.7 --yaw_rate_max 2.0 \
    --gamma_shaping 1.0 --w_potential 0.0 \
    --d_safe 0.05 --w_safety 0.0 --w_smooth 0.0 --w_action_mag 0.02 \
    --planner_horizon 15 --planner_max_vel 0.5 --planner_barrier_beta 100.0
```

### Accelerated SAC baseline

```bash
isaacrun compass_algorithm/train_compass_v4.py --headless \
    --device cuda:0 \
    --algo accel_sac \
    --seed 0 \
    --num_envs 1 \
    --total_timesteps 50_000_000 \
    --video --video_length 2_000 --video_interval 500_000 \
    --buffer_size 2_000_000 \
    --action_mode xy \
    --max_steps 8_000 \
    --goal_reward 100.0 --crash_cost 200.0 --step_cost 1.0 \
    --goal_tol 0.3 --robot_radius 0.3 \
    --leg_reach_margin 0.02 --wall_margin 0.35 \
    --vx_max 0.7 --vy_max 0.7 --yaw_rate_max 2.0 \
    --gamma_shaping 1.0 --w_potential 0.0 \
    --d_safe 0.05 --w_safety 0.0 --w_smooth 0.0 --w_action_mag 0.02 \
    --planner_horizon 15 --planner_max_vel 0.5 --planner_barrier_beta 100.0 \
    --asac_beta_0 10.0 --asac_plateau_steps 100_000 \
    --asac_decay_steps 50_000 --asac_c2_noise 0.01
```

### SAC baseline

```bash
isaacrun compass_algorithm/train_compass_v4.py --headless \
    --device cuda:0 \
    --algo sac \
    --seed 0 \
    --num_envs 1 \
    --total_timesteps 50_000_000 \
    --video --video_length 2_000 --video_interval 500_000 \
    --buffer_size 2_000_000 \
    --action_mode xy \
    --max_steps 8_000 \
    --goal_reward 100.0 --crash_cost 200.0 --step_cost 1.0 \
    --goal_tol 0.3 --robot_radius 0.3 \
    --leg_reach_margin 0.02 --wall_margin 0.35 \
    --vx_max 0.7 --vy_max 0.7 --yaw_rate_max 2.0 \
    --gamma_shaping 1.0 --w_potential 0.0 \
    --d_safe 0.05 --w_safety 0.0 --w_smooth 0.0 --w_action_mag 0.02 \
    --planner_horizon 15 --planner_max_vel 0.5 --planner_barrier_beta 100.0
```

### PPO baseline

```bash
isaacrun compass_algorithm/train_compass_v4.py --headless \
    --device cuda:0 \
    --algo ppo \
    --seed 0 \
    --num_envs 1 \
    --total_timesteps 50_000_000 \
    --video --video_length 2_000 --video_interval 500_000 \
    --buffer_size 2_000_000 \
    --action_mode xy \
    --max_steps 8_000 \
    --goal_reward 100.0 --crash_cost 200.0 --step_cost 1.0 \
    --goal_tol 0.3 --robot_radius 0.3 \
    --leg_reach_margin 0.02 --wall_margin 0.35 \
    --vx_max 0.7 --vy_max 0.7 --yaw_rate_max 2.0 \
    --gamma_shaping 1.0 --w_potential 0.0 \
    --d_safe 0.05 --w_safety 0.0 --w_smooth 0.0 --w_action_mag 0.02 \
    --planner_horizon 15 --planner_max_vel 0.5 --planner_barrier_beta 100.0
```

### HPC Cluster (SLURM)

Pre-configured `sbatch` scripts are provided and run two seeds in parallel on a single NVIDIA A40 GPU (Seed 1 is delayed by 120 s to allow Seed 0 to initialize):

```bash
sbatch sbatch_p2psac_shaped_20260117_010250/run.sh   # P2P-SAC
sbatch sbatch_accel_sac_run1                          # Accelerated SAC
sbatch sbatch_sac_run1                               # SAC
sbatch sbatch_ppo_modified_run1                      # PPO
```

---

## Hyperparameters

| Parameter | P2P-SAC | Accel. SAC |
|---|---|---|
| β₀ (initial guidance) | 10.0 | 10.0 |
| β_f (final guidance) | 10.0 | 0.0 (full decay) |
| T_p (plateau horizon) | 10⁵ steps | 10⁵ steps |
| T_d (annealing horizon) | 0 | 5×10⁴ steps |
| τ_g (gate temperature) | 1.0 | — |
| C_P (planner buffer) | 10⁶ | — |
| C (total buffer) | 2×10⁶ | 10⁶ |
| MPC horizon N | 15 | — |
| Barrier parameter β | 100 | — |
| Network | 2×256, ReLU | 2×256, ReLU |
| Learning rate | 3×10⁻⁴ (Adam) | 3×10⁻⁴ (Adam) |

> Note: setting T_d = 0 collapses the annealing phase entirely; the sole change at t = T_p is activation of the advantage gate m_φ(s), isolating its contribution.

---

## Generating Training Curve Data

To reproduce the CSV files used in the training curve figures:

```bash
python generate_data.py
```

Place the output CSVs in `results/` next to your LaTeX source. The script computes mean and ±1 std across seeds, interpolated onto a shared 76-point timestep grid, with success rate in percentage (0–100).

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{amiri2026pripg,
  author    = {Amiri, Mohsen and Amiri, Mohsen and Beikmohammadi, Ali
               and Magn{\'u}sson, Sindri and Hosseinzadeh, Mehdi},
  title     = {{PriPG-RL}: Privileged Planner-Guided Reinforcement Learning
               for Partially Observable Systems with Anytime Feasible {MPC}},
  year      = {2026},
  note      = {Preprint},
  url       = {https://github.com/mohsen1amiri/PriPG-RL_UnitreeGo2}
}
```

---

## Acknowledgements

This work was supported by the United States National Science Foundation (awards ECCS-2515358 and CNS-2502856), the Swedish Research Council (grant 2024-04058), and Sweden's Innovation Agency (Vinnova). Computational resources were provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) at C3SE, partially funded by the Swedish Research Council (grant 2022-06725).

- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) — simulation and training framework
- [Unitree Robotics](https://www.unitree.com/go2/) — Go2 quadruped platform
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) — locomotion policy
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — RL algorithm implementations

---

## License

This project is licensed under the [MIT License](LICENSE).
