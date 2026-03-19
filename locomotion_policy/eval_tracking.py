cat > "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/eval_tracking.py" <<'PY'
import argparse
import os
import numpy as np
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    # Start Isaac Sim (headless)
    app = AppLauncher({"headless": True}).app

    # Build env cfg using IsaacLab's registry helper (same idea as play.py)
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)

    # Load agent cfg entry-point from gym registry, then load cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    spec = gym.spec(args.task)
    kw = getattr(spec, "kwargs", {}) or {}
    agent_ep = (
        kw.get("rsl_rl_cfg_entry_point")
        or kw.get("agent_cfg_entry_point")
        or kw.get("cfg_entry_point")
        or kw.get("rsl_rl_runner_cfg_entry_point")
    )
    if agent_ep is None:
        raise RuntimeError(f"Could not find agent cfg entry point in gym spec kwargs. Keys: {list(kw.keys())}")

    agent_cfg = load_cfg_from_registry(agent_ep)

    # Create runner + inference policy (same core idea as play.py)
    from rsl_rl.runners import OnPolicyRunner

    log_dir = os.path.join("/tmp", "rslrl_eval")
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, agent_cfg, log_dir, device=args.device)

    # Choose checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # default for this Go2 task (what your play output shows)
        ckpt_path = ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"

    # runner.load expects a path; relative is fine because /workspace is repo root
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=args.device)

    obs, _ = env.reset()

    # We rely on the printed observation layout you saw:
    # [0:3]=base_lin_vel, [3:6]=base_ang_vel, [9:12]=velocity_commands
    errs_vx, errs_vy, errs_yaw = [], [], []
    done_count = 0

    for _ in range(args.steps):
        o = obs["policy"]
        if not torch.is_tensor(o):
            o = torch.as_tensor(o, device=args.device)

        base_lin = o[:, 0:3]
        base_ang = o[:, 3:6]
        cmd = o[:, 9:12]

        evx = (base_lin[:, 0] - cmd[:, 0]).detach().float().abs()
        evy = (base_lin[:, 1] - cmd[:, 1]).detach().float().abs()
        eyaw = (base_ang[:, 2] - cmd[:, 2]).detach().float().abs()

        errs_vx.append(evx)
        errs_vy.append(evy)
        errs_yaw.append(eyaw)

        act = policy(o)
        obs, _, terminated, truncated, _ = env.step(act)

        # crude termination counter (good enough for “does it fall often?”)
        if torch.is_tensor(terminated):
            done = (terminated | truncated).any().item()
        else:
            done = (np.asarray(terminated) | np.asarray(truncated)).any()
        if done:
            done_count += 1
            obs, _ = env.reset()

    def summarize(stack: torch.Tensor):
        x = stack.flatten().cpu().numpy()
        return float(x.mean()), float(np.sqrt((x**2).mean())), float(np.percentile(x, 95))

    vx = torch.cat(errs_vx, dim=0)
    vy = torch.cat(errs_vy, dim=0)
    yw = torch.cat(errs_yaw, dim=0)

    vx_mae, vx_rmse, vx_p95 = summarize(vx)
    vy_mae, vy_rmse, vy_p95 = summarize(vy)
    yw_mae, yw_rmse, yw_p95 = summarize(yw)

    print("\n=== Command tracking error (lower is better) ===")
    print(f"vx   : MAE={vx_mae:.3f}  RMSE={vx_rmse:.3f}  P95={vx_p95:.3f}  (m/s)")
    print(f"vy   : MAE={vy_mae:.3f}  RMSE={vy_rmse:.3f}  P95={vy_p95:.3f}  (m/s)")
    print(f"yaw  : MAE={yw_mae:.3f}  RMSE={yw_rmse:.3f}  P95={yw_p95:.3f}  (rad/s)")
    print(f"resets during eval (rough stability indicator): {done_count}\n")

    env.close()
    app.close()

if __name__ == "__main__":
    main()
PY