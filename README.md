# Go2 REAP + PPO‑MPC (Isaac Sim) — 2D Navigation with MPC‑Guided RL

This repository provides a **minimal but complete research stack** for training and evaluating **2D navigation** controllers in **NVIDIA Isaac Sim** using the **Unitree Go2** asset, while treating the robot as a **planar point‑mass / sliding body** (direct planar velocity control).

The codebase is centered around two files:

- **`go2_reap_ppo_mpc.py`**  
  Starts Isaac Sim, builds the Isaac scene + Gymnasium environment (obstacles, borders, goal), implements a **REAP‑style MPC planner** (symbolic barrier constraints), runs training, and **saves 2D trajectory plots per episode**.

- **`ppo_mpc_v1.py`**  
  Implements **`PPO_MPC`**, a PPO variant that can blend standard PPO learning with an **expert‑matching (imitation) loss**, where expert actions are produced by the REAP planner and passed through the environment `info`.

---

## Key idea

You can train an RL policy for navigation while optionally guiding learning using an MPC “expert”:

- **Agent action:** planar velocity `a = [vx, vy]`
- **Expert action (optional):** `u_expert = REAP_Planner.get_action([x, y])`
- **Training:** PPO objective (+ optional imitation to match expert actions)
- **Visualization:** episode‑level 2D plots with obstacles, borders, start/goal/end, and **dark→light time coloring** along the trajectory.

---

## Highlights

### 1) Isaac Sim + Gymnasium environment (Go2 as a “sliding” agent)
Implemented in **`Go2MPCEnv`** (`go2_reap_ppo_mpc.py`):

- **Observation:** `s = [x, y]` (world position)
- **Action:** `a = [vx, vy]` (planar velocity, clipped to `[-2, 2]`)
- **Scene:**
  - static **cylindrical obstacles** (colliders)
  - static **border walls** (colliders)
  - visual **goal marker**
- **Termination events:**
  - `goal` (within tolerance)
  - `crash_obstacle`
  - `crash_border` / `crash_oob`
  - `timeout` (episode max steps)

### 2) REAP‑style MPC expert planner (symbolic constraints)
Implemented as **`REAP_Planner`** inside `go2_reap_ppo_mpc.py`:

- Uses **SymPy** to define and compile constraints:
  - control bounds
  - obstacle avoidance constraints (barrier‑style)
- Uses a short MPC horizon and warm‑starting
- Can run as:
  - **expert labeler** during RL (expert action attached to `info["Action MPC"]`)
  - **standalone controller** for evaluation

### 3) PPO + MPC imitation (PPO_MPC)
Implemented in **`ppo_mpc_v1.py`**:

- Standard PPO training loop + an optional **expert action matching term**
- Expert actions are typically produced by the environment and passed via `info["Action MPC"]`
- Supports adaptive weighting/scheduling (e.g., “imitate more early, then rely on RL later”)

### 4) Automatic 2D episode trajectory plots (dark → light over time)
Implemented in `Go2MPCEnv._save_episode_trajectory()`:

- Draws:
  - boundary rectangle
  - obstacle circles
  - start / goal / end markers
  - trajectory points colored **dark → light** from episode start to end
- Saved automatically when an episode ends (goal/crash/timeout)

Output path (by default in this project’s training entrypoint):

```
tb_logs/<run_name>/trajectories/ep_000010_goal.png
tb_logs/<run_name>/trajectories/ep_000011_crash_obstacle.png
tb_logs/<run_name>/trajectories/ep_000012_timeout.png
```

**How often does it plot?**  
Controlled by `traj_every`:

- `traj_every = 1` → save **every episode**
- `traj_every = 5` → save every 5 episodes
- etc.

---

## Repository layout (core)

- `go2_reap_ppo_mpc.py` — environment + REAP planner + training entrypoint + trajectory plotting
- `ppo_mpc_v1.py` — PPO_MPC algorithm (PPO + optional imitation from MPC expert)
- `Buffer_Custom.py` — custom rollout buffer (stores `infos` so expert actions can be consumed during learning)

---

## Running on Isaac Sim containers (Alvis / Apptainer / Singularity)

### 0) Source the helper `.sh` file **before anything else**
This project assumes you have an environment bootstrap script (the one you shared) that:

- selects **apptainer** or **singularity**
- binds persistent cache/log/data directories
- defines helper commands:
  - `isaacpip` (install Python packages into a persistent user base)
  - `isaacpy` / `isaacrun` (run Python inside the Isaac container)

**In your terminal (from the repo root):**
```bash
source Paper_CDC.sh
# (or: source <your_script_name>.sh)
isaacnvidia   # optional sanity check
```

> The helper script must be **sourced** (not executed) so that the functions (`isaacpip`, `isaacpy`, `isaacrun`) are available in your shell.

### 1) Install Python dependencies (inside the container) using `isaacpip`
Use `isaacpip` **instead of** `pip`:

```bash
isaacpip install --user -U pip setuptools wheel
isaacpip install --user numpy torch gymnasium sympy matplotlib tensorboard stable-baselines3
```

> Depending on your Isaac Sim image, some packages may already be installed; the command above ensures consistent versions in your persistent `PYTHONUSERBASE`.

### 2) Run training using `isaacrun` (recommended)
Use `isaacrun` / `isaacpy` **instead of** `python`:

```bash
isaacrun go2_reap_ppo_mpc.py \
  --total_timesteps 6000000 \
  --seed 0 \
  --reward_mode frozenlake_cost
```

Notes:
- Isaac Sim is started by the script via `SimulationApp`.
- Headless mode is enabled by default (`SimulationApp({"headless": True})`).  
  To watch it, set `headless=False` in `go2_reap_ppo_mpc.py`.

### 3) Logs, trajectories, and model artifacts
- TensorBoard runs go under `tb_logs/` (configurable by `--log_root`)
- Trajectory images go under:
  ```
  tb_logs/<run_name>/trajectories/
  ```
- Model is saved as:
  ```
  ppo_mpc_go2_sliding_mpc_style.zip
  ```

---

## Evaluation (after training)

There are two common evaluation modes:

1) **Evaluate a trained RL policy** (PPO_MPC)
2) **Evaluate REAP MPC alone** (no learning)

### Option A — Evaluate a trained PPO_MPC policy
Create a small script, e.g. `evaluate_policy.py`, that:
- creates the environment
- loads the saved model
- runs N episodes
- relies on the environment’s built‑in trajectory plotting (already implemented)

Example:

```python
# evaluate_policy.py
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": True})

import pathlib
import numpy as np
from stable_baselines3.common.monitor import Monitor

from go2_reap_ppo_mpc import Go2MPCEnv, REAP_Planner  # if you refactor into importable module
from ppo_mpc_v1 import PPO_MPC

def main():
    env0 = Go2MPCEnv()

    # (optional) also keep the expert attached for logging/comparison
    env0.planner = REAP_Planner()

    # enable trajectory plots during evaluation
    env0.save_trajectories = True
    env0.traj_every = 1
    env0.traj_dir = pathlib.Path("eval_trajectories")

    env = Monitor(env0)

    model = PPO_MPC.load("ppo_mpc_go2_sliding_mpc_style", env=env, device="cuda")

    n_episodes = 10
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print(f"[eval] ep={ep} event={info.get('event')} reward={reward:.3f} dist={info.get('dist'):.3f}")

    env.close()

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
```

Run it with:

```bash
isaacrun evaluate_policy.py
```

> **Important:** `go2_reap_ppo_mpc.py` currently starts `SimulationApp` at import time.  
> For clean evaluation scripts, the best practice is to **refactor** the environment/planner into an importable module (no side effects on import), and keep `SimulationApp(...)` creation inside `if __name__ == "__main__":` blocks.

### Option B — Evaluate REAP MPC alone
Similarly, run the environment with REAP actions:

```python
# evaluate_reap.py
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": True})

import pathlib
from stable_baselines3.common.monitor import Monitor
from go2_reap_ppo_mpc import Go2MPCEnv, REAP_Planner  # if refactored

def main():
    env0 = Go2MPCEnv()
    planner = REAP_Planner()
    env0.save_trajectories = True
    env0.traj_every = 1
    env0.traj_dir = pathlib.Path("eval_trajectories_reap")

    env = Monitor(env0)

    n_episodes = 10
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            planner.reset()
            action = planner.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print(f"[REAP] ep={ep} event={info.get('event')} reward={reward:.3f} dist={info.get('dist'):.3f}")

    env.close()

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
```

Run it with:

```bash
isaacrun evaluate_reap.py
```

---

## Customization guide

- **Obstacles:** edit `Go2MPCEnv.OBSTACLES` (`[x, y, radius]`)  
- **Borders:** edit `x_min/x_max/y_min/y_max` in `Go2MPCEnv`
- **Trajectory plots:** adjust `_save_episode_trajectory()` and `traj_every`, `traj_dir`
- **Rewards:** switch `--reward_mode` and tune costs/weights in args or env attributes
- **MPC expert:** edit `REAP_Planner` (horizon, constraints, weights, target)

---

## Working with / extending this repo

If you want to add new planners or new RL algorithms:

- Add a new expert planner with a `get_action(obs_xy)` interface
- Attach expert actions via `info["Action MPC"]` (or implement a callable `imitation_target_fn`)
- Extend `ppo_mpc_v1.py` to change how imitation is weighted, scheduled, or combined with PPO

