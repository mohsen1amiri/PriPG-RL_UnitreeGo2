import time
import argparse
import numpy as np
import torch as th

from stable_baselines3 import SAC, PPO
from go2_sport_adapter import Go2SportAdapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, choices=["sac", "ppo"], required=True)
    p.add_argument("--model_path", type=str, required=True)

    p.add_argument("--iface", type=str, default="", help="Network interface to robot (e.g., enp3s0/eth0).")
    p.add_argument("--hz", type=float, default=50.0)

    # safety caps (start conservative!)
    p.add_argument("--vx_max", type=float, default=0.4)
    p.add_argument("--vy_max", type=float, default=0.2)
    p.add_argument("--vyaw_max", type=float, default=0.6)

    # simple smoothing (first-order)
    p.add_argument("--tau", type=float, default=0.2, help="Command smoothing time constant (s).")

    # yaw hold: keep initial yaw (helps stability if you don't control yaw in policy)
    p.add_argument("--yaw_hold_k", type=float, default=1.5)

    return p.parse_args()


def load_model(algo: str, path: str):
    if algo == "sac":
        return SAC.load(path)
    if algo == "ppo":
        return PPO.load(path)
    raise ValueError(algo)


def main():
    args = parse_args()
    dt = 1.0 / args.hz

    robot = Go2SportAdapter(net_ifname=args.iface if args.iface else None)

    if not robot.wait_for_state(timeout_s=5.0):
        raise RuntimeError("No rt/sportmodestate received. Check network interface/domain.")

    print("Standing up...")
    robot.stand_up()
    time.sleep(1.0)

    st0 = robot.get_xy_yaw()
    if st0 is None:
        raise RuntimeError("No state after stand up.")
    x0, y0, yaw0 = st0

    model = load_model(args.algo, args.model_path)

    # command filter state
    u_f = np.zeros(2, dtype=np.float32)

    print("Running policy loop. Ctrl+C to stop.")
    try:
        while True:
            st = robot.get_xy_yaw()
            if st is None:
                robot.stop_move()
                time.sleep(dt)
                continue

            x, y, yaw = st

            # --- Observation must match training: your code uses obs = [x, y]
            # On hardware, use odom-relative coordinates (subtract initial).
            obs = np.array([x - x0, y - y0], dtype=np.float32)

            # --- Policy action (vx, vy)
            action, _ = model.predict(obs, deterministic=True)

            # --- Clip to conservative limits (then we later clip to Sport limits too)
            vx = float(np.clip(action[0], -args.vx_max, args.vx_max))
            vy = float(np.clip(action[1], -args.vy_max, args.vy_max))

            # --- Smooth commands: u_f <- u_f + alpha (u - u_f)
            alpha = dt / max(args.tau, 1e-6)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            u_f[0] += alpha * (vx - u_f[0])
            u_f[1] += alpha * (vy - u_f[1])

            # --- Yaw hold (optional but recommended if policy doesn't output yaw)
            yaw_err = (yaw0 - yaw)
            # wrap to [-pi, pi]
            yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
            vyaw = float(np.clip(args.yaw_hold_k * yaw_err, -args.vyaw_max, args.vyaw_max))

            # Send world-frame command -> converted to body-frame inside adapter
            robot.send_move_world(u_f[0], u_f[1], vyaw)

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nStopping...")
        robot.stop_move()
        time.sleep(0.2)
        robot.damp()
        print("Done.")


if __name__ == "__main__":
    main()
