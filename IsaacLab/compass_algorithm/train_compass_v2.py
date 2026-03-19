import argparse
import sys
import os
import json
from datetime import datetime

# Stable Baselines 3 Imports
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
# Custom Algorithm & Environment Imports
from COMPASS_v2 import COMPASS_v2  
from compass_buffer import COMPASSBuffer
from AcceleratedSAC import AcceleratedSAC
from TrainingMetricsCallback import TrainingMetricsCallback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="COMPASS Training Script")

# --- CORE ARGS ---
parser.add_argument("--algo", type=str, default="compass_v2", choices=["compass_v2", "sac", "ppo", "accel_sac", "expert"], help="Which algorithm to train.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--total_timesteps", type=int, default=10_000_000, help="Total training steps.")

# --- VIDEO ARGS ---
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=4_000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2_000, help="Interval between video recordings.")

# --- COMPASS RL ARGS ---
parser.add_argument("--gate_tau", type=float, default=1.0, help="Temperature for the Q-Filter sigmoid.")

# ==========================================
# --- ADD THIS: ACCELERATED SAC ARGS ---
# ==========================================
parser.add_argument("--asac_beta_0", type=float, default=1.0, help="Initial weight for the imitation loss.")
parser.add_argument("--asac_plateau_steps", type=int, default=200_000, help="Steps to hold beta_e = beta_0.")
parser.add_argument("--asac_decay_steps", type=int, default=50_000, help="Steps to decay beta_e to 0 AFTER the plateau.")
parser.add_argument("--asac_c2_noise", type=float, default=0.01, help="Noise added to expert action.")

# ==========================================
# --- ADD THIS: COMPASS V2 ARGS ---
# ==========================================
parser.add_argument("--cv2_beta_0", type=float, default=10.0, help="Initial heavy weight during plateau.")
parser.add_argument("--cv2_beta_final", type=float, default=1.0, help="Final residual weight after decay.")
parser.add_argument("--cv2_plateau_steps", type=int, default=200_000, help="Steps to hold beta = beta_0.")
parser.add_argument("--cv2_decay_steps", type=int, default=500_000, help="Steps to decay beta down to beta_final.")


# --- ENVIRONMENT ARGS ---
parser.add_argument(
    "--action_mode",
    type=str,
    default="x_yaw",
    choices=["xy", "xyyaw", "x_yaw"],
    help="Action space mode: xy=(vx,vy), xyyaw=(vx,vy,yaw_rate), x_yaw=(vx,yaw_rate).")
parser.add_argument("--max_steps", type=int, default=6000, help="Max steps per episode.")
parser.add_argument("--goal_reward", type=float, default=100.0, help="Reward for reaching the goal.")
parser.add_argument("--crash_cost", type=float, default=200.0, help="Penalty for crashing.")
parser.add_argument("--step_cost", type=float, default=1.0, help="Penalty per step taken.")
parser.add_argument("--w_progress", type=float, default=50.0, help="Weight for progress reward.")
parser.add_argument("--robot_radius", type=float, default=0.3, help="Radius of the robot for math bounds.")
parser.add_argument("--leg_reach_margin", type=float, default=0.02, help="Margin for cylinders.")
parser.add_argument("--wall_margin", type=float, default=0.35, help="Margin for boundary walls.")
parser.add_argument("--goal_tol", type=float, default=0.05, help="Tolerance to trigger goal success.")

# --- PLANNER ARGS ---
parser.add_argument("--planner_horizon", type=int, default=15, help="MPC look-ahead steps.")
parser.add_argument("--planner_max_vel", type=float, default=0.25, help="Maximum velocity allowed by planner.")
parser.add_argument("--planner_barrier_beta", type=float, default=100.0, help="Repulsive force strength.")

# --- ABLATION ARGS ---


AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- IMPORTS AFTER SIMULATION APP STARTS ---
import torch
import numpy as np
from go2_env_v2 import Go2IsaacLabNavEnv_v2

try:
    from reap_planner_v2 import REAP_Planner 
except ImportError:
    REAP_Planner = None

def main():
    # =========================================================================
    # GENERATE UNIQUE RUN ID & UNIFIED OUTPUT DIRECTORY
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args_cli.algo.upper()}_Seed{args_cli.seed}_{timestamp}"
    
    # Create the MASTER folder for this specific run
    run_dir = os.path.join("./training_outputs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save the configuration JSON directly into the run folder
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args_cli), f, indent=4)

    # =========================================================================
    # ENFORCE GLOBAL SEED & PYTORCH DETERMINISM
    # =========================================================================
    set_random_seed(args_cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
        
    print(f"\n=========================================================")
    print(f"[INFO] RUN ID INITIALIZED: {run_name}")
    print(f"[INFO] Global random seed set to: {args_cli.seed}")
    print(f"[INFO] All outputs will be saved to: {run_dir}")
    print(f"=========================================================\n")

    print("[INFO] Initializing REAP Planner...")
    if REAP_Planner is not None:
        planner = REAP_Planner(
            horizon=args_cli.planner_horizon,
            max_vel=args_cli.planner_max_vel,
            barrier_beta=args_cli.planner_barrier_beta,
            robot_radius=args_cli.robot_radius
        )
    else:
        planner = None

    print("[INFO] Initializing Training Environment...")
    render_mode = "rgb_array" if args_cli.video else None


    # =========================================================================
    # BASELINE OVERRIDES
    # =========================================================================
    if args_cli.algo in ["sac", "ppo"]:
        print(f"[INFO] Baseline {args_cli.algo.upper()} selected. Forcing pure RL (beta=0).")
        planner = None # Disable planner completely
        
    elif args_cli.algo == "accel_sac":
        print(f"[INFO] Baseline ACCELERATED SAC selected. RL in control, planner logging in background.")
        # Do NOT set planner = None. We need it to populate the COMPASSBuffer!

    elif args_cli.algo == "expert":
        print(f"[INFO] Baseline PURE EXPERT selected. Planner strictly controls hardware (beta=1).")
        # Planner MUST remain active.
    
    

    train_env = Go2IsaacLabNavEnv_v2(
            planner=planner, 
            num_envs=args_cli.num_envs, 
            device=args_cli.device, 
            render_mode=render_mode,
            max_steps=args_cli.max_steps,
            goal_reward=args_cli.goal_reward,
            crash_cost=args_cli.crash_cost,
            step_cost=args_cli.step_cost,
            w_progress=args_cli.w_progress,
            robot_radius=args_cli.robot_radius,
            leg_reach_margin=args_cli.leg_reach_margin,
            wall_margin=args_cli.wall_margin,
            goal_tol=args_cli.goal_tol,
            action_mode=args_cli.action_mode,
            force_expert_control=(args_cli.algo == "expert") 
        )

    


    # --- UPDATE 1: VIDEO WRAPPER ---
    if args_cli.video:
        from gymnasium.wrappers import RecordVideo
        video_kwargs = {
            "video_folder": os.path.join(run_dir, "videos"), # <--- Route to run_dir
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        train_env = RecordVideo(train_env, **video_kwargs)

    # Route the stable-baselines monitor to the run directory
    train_env = Monitor(train_env, run_dir)
    vec_env = DummyVecEnv([lambda: train_env])

    print(f"[INFO] Initializing RL Agent: {args_cli.algo.upper()}...")

    # Shared Policy Kwargs for Orthogonal Initialization & Stability

    sac_policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(eps=1e-5)
        )

    ppo_policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(eps=1e-5)
        )



    
    if args_cli.algo == "compass_v2":
        print(f"[INFO] Initializing COMPASS_v2 (Deterministic Schedule) | Plateau: {args_cli.cv2_plateau_steps}, Decay: {args_cli.cv2_decay_steps}")
        model = COMPASS_v2(
            "MlpPolicy", 
            vec_env, 
            replay_buffer_class=COMPASSBuffer,
            replay_buffer_kwargs={},
            verbose=1, 
            seed=args_cli.seed,
            tensorboard_log=run_dir,
            device="cuda:0",
            policy_kwargs=sac_policy_kwargs,
            # Pass the new schedule parameters
            beta_0=args_cli.cv2_beta_0,
            beta_final=args_cli.cv2_beta_final,
            plateau_steps=args_cli.cv2_plateau_steps,
            decay_steps=args_cli.cv2_decay_steps,
            gate_tau=args_cli.gate_tau
        )

    elif args_cli.algo == "sac":
        model = SAC(
            "MlpPolicy", 
            vec_env, 
            verbose=1, 
            seed=args_cli.seed,
            policy_kwargs=sac_policy_kwargs,
            tensorboard_log=run_dir,
            device="cuda:0"
        )

    elif args_cli.algo == "ppo":
        model = PPO(
            "MlpPolicy", 
            vec_env, 
            n_steps=2048,
            batch_size=64,
            verbose=1, 
            policy_kwargs=ppo_policy_kwargs,
            seed=args_cli.seed,
            tensorboard_log=run_dir,
            device="cuda:0"
        )

    elif args_cli.algo == "accel_sac":
        print(f"[INFO] Initializing Accelerated SAC with Plateau: {args_cli.asac_plateau_steps}, Decay: {args_cli.asac_decay_steps}")
        model = AcceleratedSAC(
            "MlpPolicy", 
            vec_env, 
            replay_buffer_class=COMPASSBuffer,
            replay_buffer_kwargs={},
            verbose=1, 
            tensorboard_log=run_dir,
            device="cuda:0",
            seed=args_cli.seed,
            policy_kwargs=sac_policy_kwargs,
            beta_0=args_cli.asac_beta_0,                      
            plateau_steps=args_cli.asac_plateau_steps,      
            decay_steps=args_cli.asac_decay_steps,        
            c2_noise=args_cli.asac_c2_noise                     
        )

    elif args_cli.algo == "expert":
        # Pure Expert Baseline (Dummy agent that never trains)
        model = SAC(
            "MlpPolicy", 
            vec_env, 
            learning_starts=args_cli.total_timesteps + 1000, # Dynamically prevents training
            buffer_size=10_000, 
            verbose=1, 
            seed=args_cli.seed,
            tensorboard_log=run_dir,
            device="cuda:0"
        )

    # --- UPDATE 2: CALLBACKS AND LOGGING ---
    
    # 1. Create Checkpoint Callback (Saves to run_dir/checkpoints)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="model" # Simplified since it's already in the run folder
    )

    # 2. Create Custom Metrics Callback (Saves best model to run_dir)
    metrics_callback = TrainingMetricsCallback(save_path=run_dir, run_name="best_model", verbose=1)
    
    # 3. Bundle them together
    callback_list = CallbackList([checkpoint_callback, metrics_callback])

    # Set up both TensorBoard and CSV logging directly in the run folder
    new_logger = configure(run_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print("[INFO] Starting Training Loop...")
    model.learn(
        total_timesteps=args_cli.total_timesteps, 
        callback=callback_list, 
        # Don't create subfolders for TB, dump directly into run_dir
        tb_log_name="" 
    )

    # Save final model directly into the run folder
    model.save(os.path.join(run_dir, "final_model")) 
    print(f"[INFO] Training Complete. All data saved successfully to {run_dir}.")
    simulation_app.close()

if __name__ == "__main__":
    main()