# go2_reap_ppo_mpc.py

import os, pathlib
from pathlib import Path

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
import json
import pandas as pd

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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac.policies import MlpPolicy
from p2p_sac_algo_v1 import P2P_SAC, P2PReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import argparse, json
from datetime import datetime
import csv
from pathlib import Path
from go2_env_v1 import Go2REAPEnv
from reap_planner_v1 import REAP_Planner

import matplotlib
matplotlib.use("Agg")  # headless save-to-file backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle




# -------- Isaac Sim Core API imports ----------
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.objects import FixedCylinder, VisualSphere
from isaacsim.core.api.objects import FixedCuboid





def parse_args():
    p = argparse.ArgumentParser()

    # main
    p.add_argument("--mode", type=str, default="train", choices=["train", "eval", "sanity"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_root", type=str, default="tb_logs")

    # env / eval controls (keep your existing ones as needed)
    p.add_argument("--continue_on_crash", action="store_true")
    p.add_argument("--random_start", action="store_true")
    p.add_argument("--start_clearance", type=float, default=0.25)
    p.add_argument("--start_border_clearance", type=float, default=0.15)
    p.add_argument("--start_max_tries", type=int, default=200)

    # reward mode (reuse your env settings)
    p.add_argument("--reward_mode", type=str, default="frozenlake_cost", choices=["shaped", "frozenlake_cost"])
    p.add_argument("--step_cost", type=float, default=1.0)
    p.add_argument("--goal_cost", type=float, default=0.0)
    p.add_argument("--crash_cost", type=float, default=200.0)
    p.add_argument("--action_cost", type=float, default=0.01)

    # SAC hyperparams
    p.add_argument("--total_timesteps", type=int, default=6_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--train_freq", type=int, default=1)
    p.add_argument("--gradient_steps", type=int, default=1)

    # policy net
    p.add_argument("--net_layers", type=int, default=3)
    p.add_argument("--net_nodes", type=int, default=256)

    # P2P knobs
    p.add_argument("--prefill_steps", type=int, default=20_000)
    p.add_argument("--p_expert0", type=float, default=1.0)
    p.add_argument("--p_expert_end", type=float, default=0.0)
    p.add_argument("--p_expert_end_steps", type=int, default=200_000)
    p.add_argument("--label_prob", type=float, default=1.0)
    p.add_argument("--expert_noise", type=float, default=0.1)
    p.add_argument("--bc_lambda0", type=float, default=1.0)
    p.add_argument("--bc_lambda_end", type=float, default=0.0)
    p.add_argument("--bc_lambda_end_steps", type=int, default=200_000)
    p.add_argument("--qfilter_tau", type=float, default=1.0)
    p.add_argument("--bc_gradient_steps", type=int, default=1)
    p.add_argument("--qfilter_warmup_steps", type=int, default=100_000)


    # evaluation
    p.add_argument("--eval_algo", type=str, default="rl", choices=["rl", "reap"])
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument("--model_path", type=str, default="p2p_sac_go2.zip")
    p.add_argument("--deterministic", action="store_true")

    # during-training eval (reuse your callback)
    p.add_argument("--eval_freq", type=int, default=0)
    p.add_argument("--eval_n_episodes", type=int, default=20)

    # trajectory plot knobs (reuse your env logic)
    p.add_argument("--traj_every_train", type=int, default=10)
    p.add_argument("--traj_every_eval", type=int, default=10)
    p.add_argument("--save_eval_traj_during_train", action="store_true")

    # extra env knobs (you use these in main)
    p.add_argument("--goal_reward", type=float, default=100.0)
    p.add_argument("--crash_penalty", type=float, default=200.0)

    p.add_argument("--w_progress", type=float, default=50.0)
    p.add_argument("--w_time", type=float, default=0.01)
    p.add_argument("--w_u", type=float, default=0.05)
    p.add_argument("--safe_margin", type=float, default=0.15)
    p.add_argument("--w_safe", type=float, default=5.0)
    p.add_argument("--use_safety_penalty", action="store_true", default=True)
    p.add_argument("--no_safety_penalty", action="store_false", dest="use_safety_penalty")


    p.add_argument("--always_negative_reward", action="store_true")
    p.add_argument("--neg_eps", type=float, default=1e-6)


    return p.parse_args()



def build_run_name_sac(args) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return (
        f"p2psac_seed{args.seed}"
        f"_rm{args.reward_mode}"
        f"_lr{args.lr:g}_g{args.gamma:g}_tau{args.tau:g}"
        f"_buf{args.buffer_size}_bs{args.batch_size}"
        f"_pref{args.prefill_steps}"
        f"_pex0{args.p_expert0:g}_pexEnd{args.p_expert_end:g}_pexT{args.p_expert_end_steps}"
        f"_lam0{args.bc_lambda0:g}_lamEnd{args.bc_lambda_end:g}_lamT{args.bc_lambda_end_steps}"
        f"_qTau{args.qfilter_tau:g}_bcG{args.bc_gradient_steps}"
        f"_{ts}"
    )



class EpisodeEventStatsCallback(BaseCallback):
    """
    Logs:
      - success_rate: episodes that ended with goal / episodes
      - timeout_rate: episodes that ended with timeout / episodes
      - crash_end_rate: episodes that ended by crash / episodes
      - crash_any_rate: episodes that had >=1 crash event at any time / episodes
    Note: In continue_on_crash mode, crash_any_rate can be > crash_end_rate,
          and crash_any_rate + success_rate can exceed 1 (because an episode can crash and later reach goal).
    """
    def __init__(self, window_episodes=100, verbose=0):
        super().__init__(verbose)
        self.window = int(window_episodes)
        self.reset_window()

        # Per-env episode state (initialized on first step)
        self._had_crash = None

    def reset_window(self):
        self.ep = 0
        self.goal = 0
        self.timeout = 0
        self.crash_end = 0       # ended by crash
        self.crash_any = 0       # had at least one crash (even if continued)

    def _ensure_state(self, n_envs: int):
        if self._had_crash is None or len(self._had_crash) != n_envs:
            self._had_crash = [False] * n_envs

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)

        # SB3 typical case: VecEnv
        if infos is not None and dones is not None:
            self._ensure_state(len(infos))

            for i, (info, done) in enumerate(zip(infos, dones)):
                event = str(info.get("event", ""))

                # Count crash occurrence even if episode doesn't end
                if event.startswith("crash"):
                    self._had_crash[i] = True

                # Episode ended -> finalize episode-level stats
                if done:
                    self.ep += 1

                    end_event = event
                    if end_event == "goal":
                        self.goal += 1
                    elif end_event == "timeout":
                        self.timeout += 1
                    elif end_event.startswith("crash"):
                        self.crash_end += 1

                    # Count "crashed at least once this episode"
                    if self._had_crash[i]:
                        self.crash_any += 1

                    # Reset per-episode flag for this env
                    self._had_crash[i] = False

                    if self.ep >= self.window:
                        self._log_and_reset()

        else:
            # Non-vec fallback (rare in SB3)
            info = self.locals.get("info", {}) or {}
            done = bool(self.locals.get("done", False))
            event = str(info.get("event", ""))

            # Track crash even when not done
            if event.startswith("crash"):
                if self._had_crash is None:
                    self._had_crash = [False]
                self._had_crash[0] = True

            if done:
                self.ep += 1
                if event == "goal":
                    self.goal += 1
                elif event == "timeout":
                    self.timeout += 1
                elif event.startswith("crash"):
                    self.crash_end += 1

                if self._had_crash and self._had_crash[0]:
                    self.crash_any += 1
                self._had_crash[0] = False

                if self.ep >= self.window:
                    self._log_and_reset()

        return True

    def _log_and_reset(self):
        denom = max(1, self.ep)
        self.logger.record("rollout/success_rate", self.goal / denom)
        self.logger.record("rollout/timeout_rate", self.timeout / denom)
        self.logger.record("rollout/crash_end_rate", self.crash_end / denom)
        self.logger.record("rollout/crash_any_rate", self.crash_any / denom)
        self.reset_window()

    def _on_training_end(self) -> None:
        if self.ep > 0:
            denom = max(1, self.ep)
            self.logger.record("rollout/success_rate", self.goal / denom)
            self.logger.record("rollout/timeout_rate", self.timeout / denom)
            self.logger.record("rollout/crash_end_rate", self.crash_end / denom)
            self.logger.record("rollout/crash_any_rate", self.crash_any / denom)
            self.logger.dump(self.num_timesteps)





class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool = True,
        save_trajectories: bool = False,
        traj_every: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = bool(deterministic)

        self.save_trajectories = bool(save_trajectories)
        self.traj_every = int(traj_every)

        self.last_eval_step = 0
        self.out_dir = None
        self.csv_path = None
        self.traj_root = None

    def _on_training_start(self) -> None:
        # SB3 sets logger.dir when training starts
        run_dir = Path(self.model.logger.dir)
        self.out_dir = run_dir / "eval_during_train"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.traj_root = self.out_dir / "trajectories"
        self.traj_root.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.out_dir / "eval_history.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    ["timesteps", "success_rate", "crash_rate", "timeout_rate", "avg_return", "avg_len"]
                )

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True

        if (self.num_timesteps - self.last_eval_step) < self.eval_freq:
            return True

        self.last_eval_step = self.num_timesteps
        stats = self._run_eval()

        # ---- TensorBoard logging ----
        self.logger.record("eval/success_rate", stats["success_rate"])
        self.logger.record("eval/crash_rate", stats["crash_rate"])
        self.logger.record("eval/timeout_rate", stats["timeout_rate"])
        self.logger.record("eval/avg_return", stats["avg_return"])
        self.logger.record("eval/avg_len", stats["avg_len"])
        self.logger.dump(self.num_timesteps)

        # ---- CSV append ----
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [self.num_timesteps,
                 stats["success_rate"], stats["crash_rate"], stats["timeout_rate"],
                 stats["avg_return"], stats["avg_len"]]
            )

        # ---- JSON snapshot ----
        with open(self.out_dir / f"eval_{self.num_timesteps}.json", "w") as f:
            json.dump(stats, f, indent=2)

        if self.verbose:
            print(f">> [EvalDuringTrain] t={self.num_timesteps} stats={stats}")

        return True

    def _run_eval(self) -> dict:
        # If eval_env is a Monitor wrapper, unwrap one level to reach your Go2MPCEnv
        base_env = self.eval_env.env if hasattr(self.eval_env, "env") else self.eval_env

        # Configure trajectory saving for THIS eval
        if self.save_trajectories and self.traj_every > 0:
            base_env.save_trajectories = True
            base_env.traj_every = self.traj_every
            base_env.traj_dir = self.traj_root / f"t_{self.num_timesteps:010d}"
        else:
            base_env.save_trajectories = False

        n_goal = n_crash = n_timeout = 0
        returns = []
        lengths = []

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            ep_ret = 0.0
            ep_len = 0

            had_crash = False  # NEW: track crashes anywhere in the episode

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)

                ev = str(info.get("event", ""))
                if ev.startswith("crash"):
                    had_crash = True

                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_len += 1

            end_event = str(info.get("event", "unknown"))
            if end_event == "goal":
                n_goal += 1
            elif end_event == "timeout":
                n_timeout += 1

            # NEW: crash count = crash-any (works with continue_on_crash)
            if had_crash:
                n_crash += 1



            returns.append(ep_ret)
            lengths.append(ep_len)

        denom = float(self.n_eval_episodes)
        return {
            "timesteps": int(self.num_timesteps),
            "n_eval_episodes": int(self.n_eval_episodes),
            "n_goal": int(n_goal),
            "n_crash": int(n_crash),
            "n_timeout": int(n_timeout),
            "success_rate": n_goal / denom,
            "crash_rate": n_crash / denom,
            "timeout_rate": n_timeout / denom,
            "avg_return": float(np.mean(returns)) if returns else 0.0,
            "avg_len": float(np.mean(lengths)) if lengths else 0.0,
        }






def run_evaluation(args, env0, log_root: pathlib.Path):
    """
    Runs evaluation episodes and relies on env0's built-in trajectory saver.
    Saves PNGs into: log_root / eval_<algo>_<timestamp> / trajectories
    """

    eval_tag = f"eval_{args.eval_algo}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    traj_dir = log_root / eval_tag / "trajectories"

    # Make sure trajectories are ON for evaluation
    # Make sure trajectories follow CLI settings for evaluation
    env0.save_trajectories = (args.traj_every_eval > 0)
    env0.traj_every = args.traj_every_eval
    env0.traj_dir = traj_dir
    env0.random_start = args.random_start
    env0.start_clearance = args.start_clearance
    env0.start_border_clearance = args.start_border_clearance
    env0.start_max_tries = args.start_max_tries
    env0.terminate_on_crash = (not args.continue_on_crash)



    # IMPORTANT: avoid the env computing expert_u inside step() during eval
    # (it's only for imitation labels, and it's expensive)
    env0.collect_expert = False
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
        model = P2P_SAC.load(args.model_path, env=env, device="cuda")
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
    events = []
    successes = []

    for ep in range(args.eval_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0

        had_crash = False  # NEW: track crashes anywhere in the episode

        # For REAP: reset once per episode to start from a cold/warm state (your choice)
        if reap is not None:
            reap.reset()

        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            else:
                action = reap.get_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)

            ev = str(info.get("event", ""))
            if ev.startswith("crash"):
                had_crash = True

            done = bool(terminated or truncated)


            ep_ret += float(reward)
            ep_len += 1

        end_event = str(info.get("event", "unknown"))
        events.append(end_event)
        successes.append(float(info.get("is_success", 0.0)))

        if end_event == "goal":
            n_goal += 1
        elif end_event == "timeout":
            n_timeout += 1

        # NEW: crash count = crash-any (works with continue_on_crash)
        if had_crash:
            n_crash += 1


        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)

        print(f">> [Eval] ep={ep+1}/{args.eval_episodes} event={end_event} return={ep_ret:.2f} len={ep_len}")

    print(">> [Eval] Done.")
    print(f">> [Eval] success(goal)={n_goal}/{args.eval_episodes}  crash={n_crash}  timeout={n_timeout}")
    print(f">> [Eval] avg_return={np.mean(ep_returns):.2f}  avg_len={np.mean(ep_lengths):.1f}")
    print(">> [Eval] Trajectory PNGs saved in:", traj_dir)



    # ... after the evaluation loop finishes

    summary = {
        "eval_algo": args.eval_algo,
        "eval_episodes": args.eval_episodes,
        "n_goal": n_goal,
        "n_crash": n_crash,
        "n_timeout": n_timeout,
        "success_rate": n_goal / float(args.eval_episodes),
        "crash_rate": n_crash / float(args.eval_episodes),
        "timeout_rate": n_timeout / float(args.eval_episodes),
        "avg_return": float(np.mean(ep_returns)),
        "avg_len": float(np.mean(ep_lengths)),
    }

    (out_dir := (log_root / eval_tag)).mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    df = pd.DataFrame({
        "episode": np.arange(1, args.eval_episodes + 1),
        "event": events,
        "is_success": successes,
        "return": ep_returns,
        "length": ep_lengths,
    })


    df.to_csv(out_dir / "episodes.csv", index=False)

    print(">> [Eval] Saved:", out_dir / "summary.json", "and", out_dir / "episodes.csv")




# ==============================================================================
# SECTION 4: MAIN
# ==============================================================================


def main():
    print(">> [Main] Creating Go2REAPEnv (Sliding Mode)...")
    args = parse_args()
    
    set_global_seed(args.seed)
    print(f">> [Seed] Using seed = {args.seed}")
    if args.seed != EARLY_SEED:
        print(f">> [Seed WARNING] args.seed={args.seed} but EARLY_SEED={EARLY_SEED}")



    log_root = pathlib.Path(args.log_root).resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    env0 = Go2REAPEnv()

    # ---------- EVAL MODE ----------
    if args.mode == "eval":
        run_evaluation(args, env0, log_root)
        return


    # apply argparse settings to env (minimal: overwrite attributes)
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






    run_name = build_run_name_sac(args)

    # Where to save episode trajectory PNGs
    # Where to save episode trajectory PNGs (training)
    traj_dir = log_root / run_name / "trajectories"
    env0.traj_dir = traj_dir
    env0.save_trajectories = (args.traj_every_train > 0)
    env0.traj_every = args.traj_every_train
    # Random start options
    env0.random_start = args.random_start
    env0.start_clearance = args.start_clearance
    env0.start_border_clearance = args.start_border_clearance
    env0.start_max_tries = args.start_max_tries

    # Crash handling option
    env0.terminate_on_crash = (not args.continue_on_crash)




    print(">> [Main] Initialize REAP Brain...")
    planner = REAP_Planner()

 

    env0.planner = planner
    env0.collect_expert = False   # IMPORTANT: SAC queries labels via env.get_expert_action()




    env = Monitor(env0)

    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)




    def rollout(env, action_fn, max_steps=20000):
        obs, info = env.reset()
        ret = 0.0
        steps = 0
        while True:
            a = action_fn(obs)
            obs, r, terminated, truncated, info = env.step(a)
            ret += float(r)
            steps += 1
            if terminated or truncated or steps >= max_steps:
                break
        return ret, steps, info.get("event", None)

    if args.mode == "sanity":
        # 1) do-nothing
        ret0, steps0, ev0 = rollout(env, lambda obs: np.array([0.0, 0.0], dtype=np.float32))
        print(f"[SANITY] zero-action: return={ret0:.3f}, steps={steps0}, event={ev0}")

        # 2) expert warm-start (recommended)
        planner.reset()
        def expert_fn(obs):
            return planner.get_action(obs).astype(np.float32)

        retE, stepsE, evE = rollout(env, expert_fn)
        print(f"[SANITY] expert warm-start: return={retE:.3f}, steps={stepsE}, event={evE}")

        # 3) expert cold-start (optional, matches your labeling if you reset each call)
        base_env = env.unwrapped  # gets Go2REAPEnv
        def expert_cold(obs):
            return base_env.get_expert_action(obs).astype(np.float32)


        retC, stepsC, evC = rollout(env, expert_cold)
        print(f"[SANITY] expert cold-start: return={retC:.3f}, steps={stepsC}, event={evC}")

        raise SystemExit(0)





    # def imitation_target_fn(obs_batch, actions_batch, infos_batch):
    #     obs_np = obs_batch.detach().cpu().numpy()
    #     expert_actions = []
    #     for pos in obs_np:
    #         planner.reset()              # <-- IMPORTANT: reset before each query
    #         u = planner.get_action(pos)  # [vx, vy]
    #         expert_actions.append(u)
    #     return np.asarray(expert_actions, dtype=np.float32)
    policy_kwargs = dict(net_arch=[args.net_nodes] * args.net_layers)


    print(">> [Main] Creating P2P-SAC model...")


    model = P2P_SAC(
        policy=MlpPolicy,
        env=env,
        seed=args.seed,
        verbose=1,
        learning_rate=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,

        # IMPORTANT: don’t train until expert-prefill phase is done
        learning_starts=args.prefill_steps,

        # P2P pieces
        prefill_steps=args.prefill_steps,
        p_expert0=args.p_expert0,
        p_expert_end=args.p_expert_end,
        p_expert_end_steps=args.p_expert_end_steps,
        label_prob=args.label_prob,
        expert_noise=args.expert_noise,
        bc_lambda0=args.bc_lambda0,
        bc_lambda_end=args.bc_lambda_end,
        bc_lambda_end_steps=args.bc_lambda_end_steps,
        qfilter_tau=args.qfilter_tau,
        bc_gradient_steps=args.bc_gradient_steps,
        qfilter_warmup_steps=args.qfilter_warmup_steps,


        replay_buffer_class=P2PReplayBuffer,
        tensorboard_log=str(log_root),
        policy_kwargs=policy_kwargs,
        device="cuda",
    )


    print(">> [Main] Starting P2P-SAC training...")
    train_stats_cb = EpisodeEventStatsCallback(window_episodes=100)

    callbacks = [train_stats_cb]


    # ---- ONLY create eval env if you actually want eval during training ----
    if args.eval_freq > 0:
        eval_env0 = Go2REAPEnv()
        eval_env0.planner = None
        eval_env0.save_trajectories = False
        eval_env0.traj_every = args.traj_every_eval

        # copy reward/termination settings to eval env
        eval_env0.goal_reward = args.goal_reward
        eval_env0.crash_penalty = args.crash_penalty
        eval_env0.w_progress = args.w_progress
        eval_env0.w_time = args.w_time
        eval_env0.w_u = args.w_u
        eval_env0.safe_margin = args.safe_margin
        eval_env0.w_safe = args.w_safe
        eval_env0.use_safety_penalty = args.use_safety_penalty

        eval_env0.reward_mode = args.reward_mode
        eval_env0.step_cost = args.step_cost
        eval_env0.goal_cost = args.goal_cost
        eval_env0.crash_cost = args.crash_cost
        eval_env0.action_cost = args.action_cost

        eval_env0.random_start = args.random_start
        eval_env0.start_clearance = args.start_clearance
        eval_env0.start_border_clearance = args.start_border_clearance
        eval_env0.start_max_tries = args.start_max_tries
        eval_env0.terminate_on_crash = (not args.continue_on_crash)
        eval_env0.collect_expert = False
        eval_env0.planner = None    


        if getattr(args, "always_negative_reward", False):
            eval_env0._configure_always_negative_reward(eps=args.neg_eps)

        eval_env = Monitor(eval_env0)
        eval_env.reset(seed=args.seed + 999)

        


        eval_cb = PeriodicEvalCallback(
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_n_episodes,
            deterministic=args.deterministic,
            save_trajectories=args.save_eval_traj_during_train,
            traj_every=args.traj_every_eval,
            verbose=1,
        )
        callbacks.append(eval_cb)

    callback = CallbackList(callbacks)
    model.learn(total_timesteps=args.total_timesteps, tb_log_name=run_name, callback=callback)




    # Save config next to the actual TB run directory
    run_dir = pathlib.Path(model.logger.dir)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    print(">> [Main] TensorBoard run dir:", run_dir)


    print(">> [Main] Training finished, saving model...")

    save_path = Path(args.model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))   # ok if ends with .zip
    print(">> [Main] Saved model to:", save_path)

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
