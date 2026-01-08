# go2_reap_ppo_mpc.py

import os, pathlib

home = pathlib.Path.home()

# Make sure Isaac/Kit caches go to a writable place
os.environ.setdefault("XDG_CACHE_HOME", str(home / ".cache"))
os.environ.setdefault("OMNI_CACHE_ROOT", str(home / ".cache" / "ov"))
os.environ.setdefault("OMNI_KIT_CACHE_DIR", str(home / ".cache" / "ov" / "Kit"))
os.environ.setdefault("USD_CACHEDIR", str(home / ".cache" / "usd"))

# (Optional) kvdb / shaderdb – some installs use these:
os.environ.setdefault("OMNI_KVDB_ROOT", str(home / ".cache" / "ov" / "kvdb"))
os.environ.setdefault("RTX_SHADERDB_CACHE_PATH", str(home / ".cache" / "ov" / "shaderdb"))

for p in [
    home / ".cache" / "ov" / "Kit",
    home / ".cache" / "usd",
    home / ".cache" / "ov" / "kvdb",
    home / ".cache" / "ov" / "shaderdb",
]:
    p.mkdir(parents=True, exist_ok=True)

"""
Modified Minimal PPO_MPC for Unitree Go2 in Isaac Sim
Behaving like a 2D Point Mass (REAP Style)
"""

import sys
import argparse
import random
import numpy as np
import torch as th

def set_global_seed(seed: int, seed_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if seed_cuda and th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def _early_seed_from_argv(default: int = 0) -> int:
    # Parse only --seed before Isaac starts
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--seed", type=int, default=default)
    args, _ = p.parse_known_args(sys.argv[1:])
    return args.seed

EARLY_SEED = _early_seed_from_argv(0)
set_global_seed(EARLY_SEED, seed_cuda=False)
print(f">> [Seed] Early seed (before Isaac) = {EARLY_SEED}")


# -------- Start Isaac Sim headless ----------
from isaacsim.simulation_app import SimulationApp
# Toggle headless=False if you want to watch it slide!
simulation_app = SimulationApp({"headless": True})

# -------- Standard imports ----------
import gymnasium as gym
from gymnasium import spaces
import sympy as sp
from stable_baselines3.ppo.policies import MultiInputPolicy, MlpPolicy
from Buffer_Custom import RolloutBuffer, DictRolloutBuffer
from stable_baselines3.common.monitor import Monitor
import argparse, json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless save-to-file backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle




# Use our custom PPO_MPC instead of vanilla PPO
from ppo_mpc_v1 import PPO_MPC

# -------- Isaac Sim Core API imports ----------
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.objects import FixedCylinder, VisualSphere
from isaacsim.core.api.objects import FixedCuboid





# ==============================================================================
# SECTION 2: THE REAP PLANNER (The Brain)
# ==============================================================================
class REAP_Planner:
    def __init__(self):
        print(">> [REAP] Initializing Symbolic Math (this takes a moment)...")

        # --- PARAMETERS ---
        self.N_HORIZON = 2
        self.DT_SIM = 0.01
        self.MAX_VEL = 2.0
        self.BARRIER_BETA = 100.0
        self.ROBOT_RADIUS = 0.1

        # Weights
        self.Q_mat = np.diag([10.0, 10.0])
        self.R_mat = np.diag([0.5, 0.5])
        self.P_mat = np.diag([20.0, 20.0])
        self.target = np.array([0.0, 2.5])

        self.OBSTACLES = np.array([
            [0.0,   0.15, 0.43],
            [-1.3,  0.75, 0.43],
            [0.0,   1.45, 0.43],
            [1.3,   0.75, 0.43],
            [1.3,  -0.45, 0.43],
            [-1.3, -0.45, 0.43],
        ], dtype=np.float32)





        # --- SYMBOLIC MATH SETUP ---
        x_sym = sp.Matrix([sp.Symbol("x"), sp.Symbol("y")])
        u_sym = sp.Matrix([sp.Symbol("u_x"), sp.Symbol("u_y")])

        all_constraints_sym = []
        # Control Constraints
        all_constraints_sym.append(u_sym[0] - self.MAX_VEL)
        all_constraints_sym.append(-u_sym[0] - self.MAX_VEL)
        all_constraints_sym.append(u_sym[1] - self.MAX_VEL)
        all_constraints_sym.append(-u_sym[1] - self.MAX_VEL)

        # Obstacle Constraints
        for i in range(self.OBSTACLES.shape[0]):
            x_i_sym = sp.Matrix(self.OBSTACLES[i, :2])
            r_i = sp.Float(self.OBSTACLES[i, 2])
            r = sp.Float(self.ROBOT_RADIUS)

            diff_vec_sym = x_i_sym - x_sym
            dist_sq_scalar = (diff_vec_sym.T @ diff_vec_sym)[0]
            dist_sym = sp.sqrt(dist_sq_scalar)

            theta_i_sym = 0.5 - (r_i**2 - r**2) / (2 * dist_sq_scalar)
            unit_vec_robot_to_obs_sym = -diff_vec_sym / dist_sym
            b_i_term1_sym = theta_i_sym * x_i_sym + (1 - theta_i_sym) * x_sym
            b_i_term2_sym = r * unit_vec_robot_to_obs_sym
            b_i_scalar = (diff_vec_sym.T @ (b_i_term1_sym + b_i_term2_sym))[0]

            x_next_sym = x_sym + u_sym * sp.Float(self.DT_SIM)
            dot_expr = (diff_vec_sym.T @ x_next_sym)[0, 0]
            all_constraints_sym.append(dot_expr - b_i_scalar)

        self.c_per_step = len(all_constraints_sym)

        # Lambdify (Compile the math to fast functions)
        self.num_constraints_step = sp.lambdify(list(u_sym) + list(x_sym), all_constraints_sym, "numpy")
        dCdu_sym = [c_i.diff(u_var) for c_i in all_constraints_sym for u_var in u_sym]
        self.num_dCdu_step = sp.lambdify(list(u_sym) + list(x_sym), dCdu_sym, "numpy")
        dCdx_sym = [c_i.diff(x_var) for c_i in all_constraints_sym for x_var in x_sym]
        self.num_dCdx_step = sp.lambdify(list(u_sym) + list(x_sym), dCdx_sym, "numpy")

        # --- INTERNAL STATE (Warm Start) ---
        self.opt_vars = np.zeros(self.N_HORIZON * 2)
        self.hat_lambda = np.zeros(self.N_HORIZON * self.c_per_step)

        # Save a "fresh reset" copy (cold-start state)
        self._opt_vars_reset = self.opt_vars.copy()
        self._hat_lambda_reset = self.hat_lambda.copy()

    
    
    def reset(self):
        """Reset REAP warm-start so get_action() depends only on the given state."""
        self.opt_vars[:] = self._opt_vars_reset
        self.hat_lambda[:] = self._hat_lambda_reset



    def get_action(self, current_pos):
        """
        Takes current Robot Position [x, y]
        Returns Optimal Velocity [vx, vy]
        """
        for _ in range(5):
            grad_u = self._compute_grad_u(self.opt_vars, current_pos, self.hat_lambda)
            grad_l = self._compute_grad_lambda(self.opt_vars, current_pos)

            # Phi Projection
            phi_val = np.zeros_like(self.hat_lambda)
            for i in range(len(self.hat_lambda)):
                if self.hat_lambda[i] > 1e-10 or (self.hat_lambda[i] <= 1e-10 and grad_l[i] >= 0):
                    phi_val[i] = 0
                else:
                    phi_val[i] = -grad_l[i]

            checkpoint = grad_l + phi_val
            Sigma = 0.5

            # Update Primal (Control) and Dual (Lambda)
            self.opt_vars = self.opt_vars - Sigma * grad_u
            self.hat_lambda = self.hat_lambda + Sigma * checkpoint
            self.hat_lambda = np.maximum(self.hat_lambda, 0)  # Project >= 0
            self.opt_vars = np.clip(self.opt_vars, -self.MAX_VEL, self.MAX_VEL)

        # Extract first action (Model Predictive Control)
        u_applied = self.opt_vars[:2].copy()

        # Shift Warm Start
        self.opt_vars = np.roll(self.opt_vars, -2)
        self.opt_vars[-2:] = 0.0

        return u_applied

    # --- Internal Helpers ---
    def _get_trajectory(self, u_flat, x0):
        U = u_flat.reshape(self.N_HORIZON, 2)
        X = np.zeros((self.N_HORIZON + 1, 2))
        X[0] = x0
        for k in range(self.N_HORIZON):
            X[k + 1] = X[k] + U[k] * self.DT_SIM
        return U, X

    def _compute_grad_u(self, u_flat, x0, lambdas_flat):
        U, X = self._get_trajectory(u_flat, x0)
        lambdas = lambdas_flat.reshape(self.N_HORIZON, self.c_per_step)
        grad_U = np.zeros((self.N_HORIZON, 2))
        grad_x_accum = np.zeros(2)

        for k in reversed(range(self.N_HORIZON)):
            x_curr, x_next, u_curr = X[k], X[k + 1], U[k]

            grad_control = 2 * self.R_mat @ u_curr
            if k == self.N_HORIZON - 1:
                d_state_cost = 2 * self.P_mat @ (x_next - self.target)
            else:
                d_state_cost = 2 * self.Q_mat @ (x_next - self.target)

            total_grad_x_next = d_state_cost + grad_x_accum

            constraints = np.array(self.num_constraints_step(*u_curr, *x_curr)).flatten()
            dCdu = np.array(self.num_dCdu_step(*u_curr, *x_curr)).reshape(self.c_per_step, 2)
            dCdx = np.array(self.num_dCdx_step(*u_curr, *x_curr)).reshape(self.c_per_step, 2)

            grad_barrier_u = np.zeros(2)
            grad_barrier_x = np.zeros(2)

            for i in range(self.c_per_step):
                log_arg = -constraints[i]
                if log_arg < 1e-6:
                    log_arg = 1e-6
                factor = lambdas[k, i] / log_arg
                grad_barrier_u += factor * dCdu[i]
                grad_barrier_x += factor * dCdx[i]

            grad_U[k] = grad_control + total_grad_x_next * self.DT_SIM + grad_barrier_u
            grad_x_accum = total_grad_x_next + grad_barrier_x

        return grad_U.flatten()

    def _compute_grad_lambda(self, u_flat, x0):
        U, X = self._get_trajectory(u_flat, x0)
        all_grad_lambdas = []
        for k in range(self.N_HORIZON):
            c_val = np.array(self.num_constraints_step(*U[k], *X[k])).flatten()
            log_args = -self.BARRIER_BETA * c_val
            grad_l = -np.log10(np.maximum(log_args, 1e-10))
            grad_l[log_args <= 0.001] = -1e200
            all_grad_lambdas.append(grad_l)
        return np.array(all_grad_lambdas).flatten()


# ==============================================================================
# SECTION 3: THE ISAAC SIM ENVIRONMENT (The Body)
# ==============================================================================


class Go2MPCEnv(gym.Env):
    """
    Modified Environment to mimic the REAP MPC dynamics.

    - Observation: Robot World Position [x, y]
    - Action: Robot Body Velocity [v_x, v_y]
    - Physics: "Sliding" (Direct velocity control, ignoring legs)
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        print(">> [Env] Creating World and loading Go2...")
        self.world = World(backend="numpy")
        self.world.scene.add_default_ground_plane()

        assets_root = get_assets_root_path()
        go2_usd = assets_root + "/Isaac/Robots/Unitree/Go2/go2.usd"

        self.robot_prim_path = "/World/Go2"
        add_reference_to_stage(usd_path=go2_usd, prim_path=self.robot_prim_path)

        self.robot = SingleArticulation(prim_path=self.robot_prim_path, name="go2")
        self.world.scene.add(self.robot)

        # Expert planner (set from main). If None, no expert labels are produced.
        self.planner = None


        # Reset to load physics handles
        self.world.reset()
        self.robot.initialize()

        # Target from REAP code
        self.target_pos = np.array([0.0, 2.5], dtype=np.float32)


        # --- Reward / termination params ---
        self.DT_SIM = 0.01                 # used for scaling only
        self.goal_tol = 0.1
        self.goal_reward = 100.0

        self.alive_cost = 1.0              # makes every non-goal step strictly negative
        self.w_dist = 10.0                  # distance^2 weight
        self.w_step = 0.5                  # step-length cost weight (via ||u||^2)

        # --- New dense shaping reward (progress-based) ---
        self.w_progress = 50.0     # reward for reducing distance to goal
        self.w_time = 0.01         # per-step time penalty (shortest-time)
        self.w_u = 0.05            # action effort penalty weight (||u||^2)

        # Optional: smooth safety shaping (penalize getting too close before collision)
        self.use_safety_penalty = True
        self.safe_margin = 0.15    # meters outside obstacle boundary
        self.w_safe = 5.0          # weight for near-obstacle penalty

        # --- Always-negative reward mode ---
        self.always_negative_reward = False
        self.neg_eps = 1e-6   # reward will be <= -neg_eps



        self.crash_penalty = 200.0         # obstacle/boundary hit penalty

        # Obstacles (use same format as REAP: [x, y, radius])
        self.robot_radius = 0.10
        self.OBSTACLES = np.array([
            [0.0,   0.15, 0.43],
            [-1.3,  0.75, 0.43],
            [0.0,   1.45, 0.43],
            [1.3,   0.75, 0.43],
            [1.3,  -0.45, 0.43],
            [-1.3, -0.45, 0.43],
        ], dtype=np.float32)

        # ----------------------------
        # Spawn REAL obstacles in Isaac Sim (static colliders)
        # ----------------------------
        self.obstacle_height = 1.0
        self.obstacle_prims = []

        for i, (ox, oy, r) in enumerate(self.OBSTACLES):
            prim_path = f"/World/Obstacles/obs_{i}"
            cyl = FixedCylinder(
                prim_path=prim_path,
                name=f"obs_{i}",  # <<< IMPORTANT: unique name
                position=np.array([float(ox), float(oy), self.obstacle_height * 0.5]),
                radius=float(r),
                height=float(self.obstacle_height),
                color=np.array([1.0, 0.2, 0.2]),
            )
            self.world.scene.add(cyl)
            self.obstacle_prims.append(cyl)

        # ----------------------------
        # Visual-only goal marker (no collisions)
        # ----------------------------
        self.goal_marker = VisualSphere(
            prim_path="/World/Goal",
            name="goal",  # <<< IMPORTANT: unique name
            position=np.array([float(self.target_pos[0]), float(self.target_pos[1]), 0.1]),
            radius=0.12,
            color=np.array([0.2, 1.0, 0.2]),
        )
        self.world.scene.add(self.goal_marker)




        # Boundaries (pick something that matches your map)
        self.x_min, self.x_max = -2.0, 2.0
        self.y_min, self.y_max = -2.0, 3.5

        # ----------------------------
        # Spawn REAL border walls in Isaac Sim (static colliders)
        # ----------------------------
        self.wall_thickness = 0.1
        wall_thickness = self.wall_thickness    

        wall_height = 1.0
        z = wall_height * 0.5

        xmin, xmax = float(self.x_min), float(self.x_max)
        ymin, ymax = float(self.y_min), float(self.y_max)

        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)

        xlen = (xmax - xmin) + 2 * wall_thickness
        ylen = (ymax - ymin) + 2 * wall_thickness

        self.border_walls = []

        # Bottom wall (y = ymin)
        self.border_walls.append(
            FixedCuboid(
                prim_path="/World/Borders/bottom",
                name="border_bottom",
                position=np.array([xmid, ymin - wall_thickness * 0.5, z]),
                scale=np.array([xlen, wall_thickness, wall_height]),
                color=np.array([0.2, 0.2, 1.0]),
            )
        )

        # Top wall (y = ymax)
        self.border_walls.append(
            FixedCuboid(
                prim_path="/World/Borders/top",
                name="border_top",
                position=np.array([xmid, ymax + wall_thickness * 0.5, z]),
                scale=np.array([xlen, wall_thickness, wall_height]),
                color=np.array([0.2, 0.2, 1.0]),
            )
        )

        # Left wall (x = xmin)
        self.border_walls.append(
            FixedCuboid(
                prim_path="/World/Borders/left",
                name="border_left",
                position=np.array([xmin - wall_thickness * 0.5, ymid, z]),
                scale=np.array([wall_thickness, ylen, wall_height]),
                color=np.array([0.2, 0.2, 1.0]),
            )
        )

        # Right wall (x = xmax)
        self.border_walls.append(
            FixedCuboid(
                prim_path="/World/Borders/right",
                name="border_right",
                position=np.array([xmax + wall_thickness * 0.5, ymid, z]),
                scale=np.array([wall_thickness, ylen, wall_height]),
                color=np.array([0.2, 0.2, 1.0]),
            )
        )

        for w in self.border_walls:
            self.world.scene.add(w)


        # Optional: episode timeout
        self.max_steps = int(90.0 / self.DT_SIM)   # ~9000
        self.step_count = 0


        # Action: velocity [vx, vy]
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)

        # Observation: position [x, y]
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(2,), dtype=np.float32)

        print(f">> [Env] Action Space: {self.action_space}")
        print(f">> [Env] Observation Space: {self.observation_space}")

        # --- Reward mode switch ---
        # "shaped" = your current progress-based reward
        # "frozenlake_cost" = simple cost-based reward (Option A)
        self.reward_mode = "frozenlake_cost"

        # --- FrozenLake-inspired (cost) reward params ---
        self.step_cost = 1.0       # reward per step = -step_cost
        self.goal_cost = 0.0       # reward at goal = -goal_cost (0 means 0; will be clamped if always_negative_reward)
        self.crash_cost = 200.0    # reward at crash = -crash_cost
        self.action_cost = 0.01    # extra penalty: -action_cost * ||u||^2



        # ----------------------------
        # Trajectory logging (2D plots)
        # ----------------------------
        self.save_trajectories = True          # master switch
        self.traj_every = 1                    # save every N episodes (1 = every episode)
        self.traj_dir = pathlib.Path("trajectories")  # will be overwritten from main()
        self._episode_idx = 0
        self._traj_xy = []                     # list of (x,y) during episode


    



    # ------------- Helpers -------------





    def _hit_border(self, obs_xy: np.ndarray):
        """
        Returns (hit: bool, side: str|None)
        side in {"left","right","bottom","top"}.
        """
        x, y = float(obs_xy[0]), float(obs_xy[1])

        # Inner faces of your walls are exactly at x_min/x_max/y_min/y_max
        # Robot "hits" border when its center comes within robot_radius of that face.
        margin = float(self.robot_radius) + 1e-3

        if x <= self.x_min + margin:
            return True, "left"
        if x >= self.x_max - margin:
            return True, "right"
        if y <= self.y_min + margin:
            return True, "bottom"
        if y >= self.y_max - margin:
            return True, "top"

        return False, None


    def _max_step_positive_reward(self) -> float:
        """
        Upper bound on the *most positive* part of:
        w_progress * progress - w_u * ||u||^2
        given action limits and DT_SIM.
        This lets us choose w_time so reward is always negative.
        """
        # action_space.high is [2,2] so vmax = sqrt(2^2+2^2)=2*sqrt(2)
        vmax = float(np.linalg.norm(self.action_space.high))
        A = float(self.w_progress) * float(self.DT_SIM)  # progress <= v*DT

        # Maximize f(v)=A*v - w_u*v^2 on v in [0, vmax]
        if self.w_u <= 0:
            return A * vmax

        v_star = A / (2.0 * float(self.w_u))
        if v_star <= vmax:
            return (A * A) / (4.0 * float(self.w_u))

        return A * vmax - float(self.w_u) * (vmax * vmax)


    def _configure_always_negative_reward(self, eps: float = 1e-6) -> None:
        """
        Turns on always-negative reward.
        For shaped mode: adjust goal_reward and w_time.
        For frozenlake_cost mode: ensure costs are >= eps so rewards are strictly negative.
        """
        self.always_negative_reward = True
        self.neg_eps = float(eps)

        if self.reward_mode == "shaped":
            # 1) goal reward must be negative
            if self.goal_reward >= -self.neg_eps:
                self.goal_reward = -self.neg_eps

            # 2) ensure normal-step reward is always < 0
            fmax = self._max_step_positive_reward()
            if self.w_time <= fmax + self.neg_eps:
                self.w_time = fmax + self.neg_eps

        else:  # frozenlake_cost
            # Make all costs at least eps so rewards are strictly negative even without clamping.
            if self.step_cost <= self.neg_eps:
                self.step_cost = self.neg_eps
            if self.goal_cost <= self.neg_eps:
                self.goal_cost = self.neg_eps
            if self.crash_cost <= self.neg_eps:
                self.crash_cost = self.neg_eps



    def _out_of_bounds(self, obs_xy: np.ndarray) -> bool:
        x, y = float(obs_xy[0]), float(obs_xy[1])
        return (x < self.x_min) or (x > self.x_max) or (y < self.y_min) or (y > self.y_max)


    def _hit_obstacle(self, obs_xy: np.ndarray) -> bool:
        p = obs_xy.astype(np.float32)
        for ox, oy, r in self.OBSTACLES:
            if np.linalg.norm(p - np.array([ox, oy], dtype=np.float32)) <= float(r + self.robot_radius):
                return True
        return False


    def _get_obs(self) -> np.ndarray:
        pos, orient = self.robot.get_world_pose()
        obs = np.array([pos[0], pos[1]], dtype=np.float32)
        return obs


    def _min_obstacle_distance(self, obs_xy: np.ndarray) -> float:
        """
        Minimum signed distance to obstacle boundary.
        Positive => outside (safe), Negative => inside (collision).
        """
        p = obs_xy.astype(np.float32)
        dmin = float("inf")
        for ox, oy, r in self.OBSTACLES:
            d = float(np.linalg.norm(p - np.array([ox, oy], dtype=np.float32)) - (float(r) + self.robot_radius))
            dmin = min(dmin, d)
        return dmin

    def _save_episode_trajectory(self, event: str) -> None:
        """
        Save a 2D plot of the episode:
        - obstacles (circles)
        - boundary box (rectangle)
        - trajectory points colored dark->light over time
        - start / goal / end markers
        """
        if not self.save_trajectories:
            return
        if self._episode_idx % self.traj_every != 0:
            return
        if len(self._traj_xy) < 2:
            return

        self.traj_dir.mkdir(parents=True, exist_ok=True)

        traj = np.asarray(self._traj_xy, dtype=np.float32)  # (T,2)
        t = np.linspace(0.0, 1.0, traj.shape[0], dtype=np.float32)  # 0=start, 1=end

        fig, ax = plt.subplots(figsize=(7, 7))

        # ---- Draw boundaries as a rectangle ----
        ax.add_patch(
            Rectangle(
                (self.x_min, self.y_min),
                self.x_max - self.x_min,
                self.y_max - self.y_min,
                fill=False,
                linewidth=2,
            )
        )

        # ---- Draw obstacles as circles ----
        for (ox, oy, r) in self.OBSTACLES:
            ax.add_patch(Circle((float(ox), float(oy)), float(r), fill=False, linewidth=2))

        # ---- Trajectory: faint line + time-colored points (dark->light) ----
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1, alpha=0.25)

        ax.scatter(
            traj[:, 0], traj[:, 1],
            c=t,
            cmap="Greys_r",   # start dark, end light
            s=12,
            linewidths=0
        )

        # ---- Start / Goal / End ----
        ax.scatter(traj[0, 0], traj[0, 1], marker="*", s=140, label="start")
        ax.scatter(self.target_pos[0], self.target_pos[1], marker="o", s=90, label="goal")
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=90, label="end")

        # ---- Plot formatting ----
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.x_min - 0.2, self.x_max + 0.2)
        ax.set_ylim(self.y_min - 0.2, self.y_max + 0.2)
        ax.set_title(f"Episode {self._episode_idx} | event={event} | steps={traj.shape[0]}")
        ax.legend(loc="upper right")

        out = self.traj_dir / f"ep_{self._episode_idx:06d}_{event}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)



    @staticmethod
    def parse_args():
        p = argparse.ArgumentParser()
        # training
        p.add_argument("--total_timesteps", type=int, default=6_000_000)
        p.add_argument("--lr", type=float, default=1e-4)
        p.add_argument("--gamma", type=float, default=0.98)
        p.add_argument("--n_steps", type=int, default=2048)
        p.add_argument("--batch_size", type=int, default=128)
        p.add_argument("--n_epochs", type=int, default=100)
        p.add_argument("--max_grad_norm", type=float, default=10.0)

        # imitation
        p.add_argument("--mse_weight", type=float, default=1.0)
        p.add_argument("--initial_beta", type=float, default=1.0)
        p.add_argument("--end_iteration_number", type=int, default=100_000)

        # env reward knobs
        p.add_argument("--w_dist", type=float, default=10.0)
        p.add_argument("--w_step", type=float, default=0.5)
        p.add_argument("--alive_cost", type=float, default=1.0)
        p.add_argument("--goal_reward", type=float, default=100.0)
        p.add_argument("--crash_penalty", type=float, default=200.0)

        p.add_argument("--w_progress", type=float, default=50.0)
        p.add_argument("--w_time", type=float, default=0.01)
        p.add_argument("--w_u", type=float, default=0.05)
        p.add_argument("--safe_margin", type=float, default=0.15)
        p.add_argument("--w_safe", type=float, default=5.0)
        p.add_argument("--use_safety_penalty", action="store_true")
        p.add_argument("--always_negative_reward", action="store_true",
               help="Force all rewards to be strictly negative")
        p.add_argument("--neg_eps", type=float, default=1e-6,
                    help="Tiny negative epsilon used when clamping")

        # policy network
        p.add_argument("--net_layers", type=int, default=3)
        p.add_argument("--net_nodes", type=int, default=256)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument(
            "--reward_mode",
            type=str,
            default="frozenlake_cost",
            choices=["shaped", "frozenlake_cost"],
            help="Choose reward function: shaped (old) or frozenlake_cost (simple cost).",
        )

        # FrozenLake-cost params
        p.add_argument("--step_cost", type=float, default=1.0)
        p.add_argument("--goal_cost", type=float, default=0.0)
        p.add_argument("--crash_cost", type=float, default=200.0)
        p.add_argument("--action_cost", type=float, default=0.01)

        # ----------------------------
        # evaluation
        # ----------------------------
        p.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
        p.add_argument("--eval_algo", type=str, default="rl", choices=["rl", "reap"])
        p.add_argument("--eval_episodes", type=int, default=20)
        p.add_argument("--model_path", type=str, default="ppo_mpc_go2_sliding_mpc_style.zip")
        p.add_argument("--deterministic", action="store_true")






        # logging
        p.add_argument("--log_root", type=str, default="tb_logs")
        return p.parse_args()

    @staticmethod
    def build_run_name(args) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        pid = os.getpid()

        # If you implemented the "default ON" safety switch:
        #   p.add_argument("--no_safety_penalty", action="store_true", ...)
        safe_flag = int(args.use_safety_penalty)

        base = (
            f"go2reap"
            f"_seed{args.seed}"
            f"_rm{args.reward_mode}"
            f"_lr{args.lr:g}_g{args.gamma:g}"
            f"_ns{args.n_steps}_bs{args.batch_size}_ne{args.n_epochs}"
            f"_mgn{args.max_grad_norm:g}"
            f"_mw{args.mse_weight:g}_b0{args.initial_beta:g}_end{args.end_iteration_number}"
            f"_safe{safe_flag}"
            f"_L{args.net_layers}_H{args.net_nodes}"
        )

        if args.reward_mode == "shaped":
            base += (
                f"_GR{args.goal_reward:g}_CP{args.crash_penalty:g}"
                f"_wp{args.w_progress:g}_wt{args.w_time:g}_wu{args.w_u:g}"
                f"_sm{args.safe_margin:g}_ws{args.w_safe:g}"
            )
        else:  # frozenlake_cost
            base += (
                f"_SC{args.step_cost:g}_GC{args.goal_cost:g}"
                f"_CC{args.crash_cost:g}_AC{args.action_cost:g}"
                f"_sm{args.safe_margin:g}_ws{args.w_safe:g}"
            )

        return f"{base}_{ts}_pid{pid}"





    # ------------- Gym API -------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print(">> [Env] reset() called")

        self.step_count = 0

        self.world.reset()
        self.robot.initialize()

        # Start position
        self.robot.set_world_pose(position=np.array([-0.01, -1.5, 0.35]))


        # Stop any movement
        self.robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self.robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

        # Settle
        for _ in range(10):
            self.world.step(render=False)

        obs = self._get_obs()
         # Start a new trajectory buffer
        self._episode_idx += 1
        self._traj_xy = [obs.copy()]

        self.prev_dist = float(np.linalg.norm(obs - self.target_pos))
        if self.use_safety_penalty:
            self.prev_dmin = self._min_obstacle_distance(obs)
        return obs, {}


    def step(self, action):
        # ---- Expert label for current state s_t (before applying agent action) ----
        obs_t = self._get_obs()
        expert_u = None
        if self.planner is not None:
            self.planner.reset()                 # stateless expert per query
            expert_u = self.planner.get_action(obs_t).copy()  # shape (2,)

        vx, vy = np.clip(action, -2.0, 2.0)


        velocity_command = np.array([vx, vy, 0.0])
        self.robot.set_linear_velocity(velocity_command)

        # Step physics
        self.world.step(render=False)

        # Observation
        obs = self._get_obs()
        # Record trajectory point
        self._traj_xy.append(obs.copy())


        # ---- Reward & termination (new) ----
        self.step_count += 1

        dist = np.linalg.norm(obs - self.target_pos)

        # 1) Goal: only time reward is non-negative
        # 1) Goal
        if dist < self.goal_tol:
            if self.reward_mode == "shaped":
                reward = self.goal_reward
            else:
                reward = -self.goal_cost

            if self.always_negative_reward and reward >= -self.neg_eps:
                reward = -self.neg_eps

            terminated = True
            truncated = False
            info = {"event": "goal", "dist": float(dist)}
            if expert_u is not None:
                info["Action MPC"] = expert_u
            
            self._save_episode_trajectory(event="goal")
            return obs, reward, terminated, truncated, info




        # 2) Crash: obstacles or boundaries => big negative reward
        hit_obs = self._hit_obstacle(obs)
        hit_border, border_side = self._hit_border(obs)

        # (Optional failsafe) if something weird happens and it escapes the box
        out = self._out_of_bounds(obs)

        crash = hit_obs or hit_border or out
        if crash:
            if self.reward_mode == "shaped":
                reward = -self.crash_penalty
            else:
                reward = -self.crash_cost

            terminated = True
            truncated = False

            # Signal what happened
            if hit_obs:
                event = "crash_obstacle"
            elif hit_border:
                event = "crash_border"
            else:
                event = "crash_oob"

            info = {
                "event": event,
                "dist": float(dist),
                "hit_obstacle": bool(hit_obs),
                "hit_border": bool(hit_border),
                "border_side": border_side,   # left/right/bottom/top or None
            }
            if expert_u is not None:
                info["Action MPC"] = expert_u
            
            self._save_episode_trajectory(event=event)
            return obs, reward, terminated, truncated, info




        # 3) Normal step
        u = np.array([vx, vy], dtype=np.float32)

        if self.reward_mode == "shaped":
            # progress toward goal (positive if getting closer)
            progress = float(self.prev_dist - dist)
            self.prev_dist = float(dist)

            reward = (self.w_progress * progress) - (self.w_u * float(u @ u)) - self.w_time
        else:
            # FrozenLake-inspired cost:
            reward = -self.step_cost - self.action_cost * float(u @ u)
            # reward = -self.step_cost 

        # Optional: near-obstacle shaping (works for both modes)
        if self.use_safety_penalty:
            dmin = self._min_obstacle_distance(obs)
            if dmin < self.safe_margin:
                reward -= self.w_safe * float((self.safe_margin - dmin) ** 2)

        # Force strictly negative if requested
        if self.always_negative_reward and reward >= -self.neg_eps:
            reward = -self.neg_eps





        terminated = False
        truncated = (self.step_count >= self.max_steps)
        info = {"event": "step", "dist": float(dist)}
        if self.reward_mode == "shaped":
            info["progress"] = progress
        if expert_u is not None:
            info["Action MPC"] = expert_u

        if truncated:
            event = "timeout"
            if self.reward_mode == "shaped":
                reward = reward  # keep whatever you already computed
            else:
                reward = reward

            info["event"] = event
            self._save_episode_trajectory(event=event)
            return obs, reward, False, True, info

        return obs, reward, terminated, truncated, info




def run_evaluation(args, env0, log_root: pathlib.Path):
    """
    Runs evaluation episodes and relies on env0's built-in trajectory saver.
    Saves PNGs into: log_root / eval_<algo>_<timestamp> / trajectories
    """

    eval_tag = f"eval_{args.eval_algo}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    traj_dir = log_root / eval_tag / "trajectories"

    # Make sure trajectories are ON for evaluation
    env0.save_trajectories = True
    env0.traj_every = 1
    env0.traj_dir = traj_dir

    # IMPORTANT: avoid the env computing expert_u inside step() during eval
    # (it's only for imitation labels, and it's expensive)
    env0.planner = None

    env = Monitor(env0)
    env.reset(seed=args.seed)

    # ----------------------------
    # Choose evaluation controller
    # ----------------------------
    model = None
    reap = None

    if args.eval_algo == "rl":
        # Load your trained model
        # If PPO_MPC inherits SB3 BaseAlgorithm, this works:
        model = PPO_MPC.load(args.model_path, env=env, device="cuda")
        print(f">> [Eval] Loaded RL model from: {args.model_path}")
    else:
        reap = REAP_Planner()
        print(">> [Eval] Using REAP controller")

    # ----------------------------
    # Rollouts
    # ----------------------------
    n_goal = 0
    n_crash = 0
    n_timeout = 0
    ep_returns = []
    ep_lengths = []

    for ep in range(args.eval_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0

        # For REAP: reset once per episode to start from a cold/warm state (your choice)
        if reap is not None:
            reap.reset()

        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            else:
                action = reap.get_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            ep_ret += float(reward)
            ep_len += 1

        # env saved the PNG already (goal/crash/timeout)
        event = info.get("event", "unknown")
        if event == "goal":
            n_goal += 1
        elif event.startswith("crash"):
            n_crash += 1
        elif event == "timeout":
            n_timeout += 1

        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)

        print(f">> [Eval] ep={ep+1}/{args.eval_episodes} event={event} return={ep_ret:.2f} len={ep_len}")

    print(">> [Eval] Done.")
    print(f">> [Eval] success(goal)={n_goal}/{args.eval_episodes}  crash={n_crash}  timeout={n_timeout}")
    print(f">> [Eval] avg_return={np.mean(ep_returns):.2f}  avg_len={np.mean(ep_lengths):.1f}")
    print(">> [Eval] Trajectory PNGs saved in:", traj_dir)



# ==============================================================================
# SECTION 4: MAIN
# ==============================================================================


def main():
    print(">> [Main] Creating Go2MPCEnv (Sliding Mode)...")
    args = Go2MPCEnv.parse_args()
    
    set_global_seed(args.seed)
    print(f">> [Seed] Using seed = {args.seed}")
    if args.seed != EARLY_SEED:
        print(f">> [Seed WARNING] args.seed={args.seed} but EARLY_SEED={EARLY_SEED}")



    log_root = pathlib.Path(args.log_root).resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    env0 = Go2MPCEnv()

    # ---------- EVAL MODE ----------
    if args.mode == "eval":
        run_evaluation(args, env0, log_root)
        return


    # apply argparse settings to env (minimal: overwrite attributes)
    env0.w_dist = args.w_dist
    env0.w_step = args.w_step
    env0.alive_cost = args.alive_cost
    env0.goal_reward = args.goal_reward
    env0.crash_penalty = args.crash_penalty
    env0.w_progress = args.w_progress
    env0.w_time = args.w_time
    env0.w_u = args.w_u
    env0.safe_margin = args.safe_margin
    env0.w_safe = args.w_safe
    env0.use_safety_penalty = args.use_safety_penalty
    # Reward mode + FrozenLake-cost params
    env0.reward_mode = args.reward_mode
    env0.step_cost = args.step_cost
    env0.goal_cost = args.goal_cost
    env0.crash_cost = args.crash_cost
    env0.action_cost = args.action_cost


    if getattr(args, "always_negative_reward", False):
        env0._configure_always_negative_reward(eps=args.neg_eps)

        if args.reward_mode == "shaped":
            args.goal_reward = env0.goal_reward
            args.w_time = env0.w_time
            print(
                ">> [Env] always_negative_reward=ON | mode=shaped "
                f"| goal_reward={env0.goal_reward:g} | w_time={env0.w_time:g} "
                f"| w_progress={env0.w_progress:g} | w_u={env0.w_u:g}"
            )
        else:
            args.step_cost = env0.step_cost
            args.goal_cost = env0.goal_cost
            args.crash_cost = env0.crash_cost
            print(
                ">> [Env] always_negative_reward=ON | mode=frozenlake_cost "
                f"| step_cost={env0.step_cost:g} | goal_cost={env0.goal_cost:g} "
                f"| crash_cost={env0.crash_cost:g} | action_cost={env0.action_cost:g}"
            )






    run_name = Go2MPCEnv.build_run_name(args)

    # Where to save episode trajectory PNGs
    traj_dir = log_root / run_name / "trajectories"
    env0.traj_dir = traj_dir
    env0.save_trajectories = True
    env0.traj_every = 1   # change to 5 or 10 if you want fewer plots


    print(">> [Main] Initialize REAP Brain...")
    planner = REAP_Planner()
    env0.planner = planner            # <-- ADD THIS LINE

    env = Monitor(env0)

    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)





    # def imitation_target_fn(obs_batch, actions_batch, infos_batch):
    #     obs_np = obs_batch.detach().cpu().numpy()
    #     expert_actions = []
    #     for pos in obs_np:
    #         planner.reset()              # <-- IMPORTANT: reset before each query
    #         u = planner.get_action(pos)  # [vx, vy]
    #         expert_actions.append(u)
    #     return np.asarray(expert_actions, dtype=np.float32)
    policy_kwargs = dict(net_arch=[args.net_nodes] * args.net_layers)


    print(">> [Main] Creating PPO_MPC model...")
    model = PPO_MPC(
        policy=MlpPolicy,
        env=env,
        seed=args.seed,
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        n_epochs=args.n_epochs,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=str(log_root),   # <-- TB root folder
        rollout_buffer_class=RolloutBuffer,
        device="cuda",
        policy_kwargs=policy_kwargs,     # <-- ADD THIS
        Adaptive_Beta=True,
        Just_Beta=False,
        MSE_Weight=args.mse_weight,
        Initial_Beta=args.initial_beta,
        start_iteration_number=0,
        end_iteration_number=args.end_iteration_number,
        # imitation_target_fn=imitation_target_fn,
    )


    print(">> [Main] Starting PPO_MPC training...")
    model.learn(total_timesteps=args.total_timesteps, tb_log_name=run_name)

    # Save config next to the actual TB run directory
    run_dir = pathlib.Path(model.logger.dir)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    print(">> [Main] TensorBoard run dir:", run_dir)


    print(">> [Main] Training finished, saving model...")
    model.save("ppo_mpc_go2_sliding_mpc_style")
    print(">> [Main] Done.")

    # # Optional: Evaluate REAP controller alone
    # print(">> [Main] Evaluating REAP controller in env...")
    # obs, _ = env.reset()

    # for t in range(1000):
    #     current_pos = obs
    #     action_vel = planner.get_action(current_pos)
    #     obs, reward, terminated, truncated, info = env.step(action_vel)

    #     dist = np.linalg.norm(current_pos - planner.target)
    #     if dist < 0.1 or terminated or truncated:
    #         print(f">> GOAL REACHED at step {t}, dist={dist:.3f}")
    #         break


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(">> [PYTHON ERROR] Unhandled exception in main():", repr(e))
        traceback.print_exc()
    finally:
        simulation_app.close()
