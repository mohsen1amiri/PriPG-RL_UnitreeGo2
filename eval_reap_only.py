#!/usr/bin/env python3
import argparse
import numpy as np

# Import your module (this will start SimulationApp because it's at top-level in your file)
import go2_reap_ppo_mpc as m


def run_episodes(
    episodes: int,
    seed: int,
    mode: str,
    reward_mode: str = "frozenlake_cost",
    step_cost: float = 1.0,
    goal_cost: float = 0.0,
    crash_cost: float = 200.0,
    action_cost: float = 0.0,
):
    """
    mode:
      - "stateful": planner.reset() once per episode (true warm-start MPC)
      - "stateless": planner.reset() every step (matches your imitation label style)
    """
    assert mode in ("stateful", "stateless")

    env = m.Go2MPCEnv()
    env.planner = None  # IMPORTANT: prevent env.step() from resetting our planner

    # Set reward mode (optional; doesn't change termination, but changes returns)
    env.reward_mode = reward_mode
    env.step_cost = float(step_cost)
    env.goal_cost = float(goal_cost)
    env.crash_cost = float(crash_cost)
    env.action_cost = float(action_cost)

    planner = m.REAP_Planner()

    lengths = []
    returns = []
    outcomes = {"goal": 0, "crash": 0, "timeout": 0}

    # Make results deterministic-ish
    np.random.seed(seed)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)  # different reset seed per episode
        planner.reset()

        ep_ret = 0.0
        ep_len = 0
        outcome = "timeout"

        while True:
            if mode == "stateless":
                planner.reset()

            action = planner.get_action(obs)  # REAP outputs [vx, vy]
            obs, r, terminated, truncated, info = env.step(action)

            ep_ret += float(r)
            ep_len += 1

            if terminated or truncated:
                # info["event"] is "goal" / "crash" / "step"
                # if truncated==True, it ended by timeout
                if terminated:
                    outcome = info.get("event", "unknown")
                    if outcome not in ("goal", "crash"):
                        outcome = "unknown"
                else:
                    outcome = "timeout"
                break

        lengths.append(ep_len)
        returns.append(ep_ret)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    lengths = np.asarray(lengths, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)

    dt = float(env.DT_SIM)  # 0.01
    report = {
        "mode": mode,
        "episodes": episodes,
        "seed": seed,
        "avg_len_steps": float(lengths.mean()),
        "std_len_steps": float(lengths.std(ddof=1)) if episodes > 1 else 0.0,
        "avg_len_seconds": float((lengths * dt).mean()),
        "avg_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=1)) if episodes > 1 else 0.0,
        "outcomes": outcomes,
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)

    # Evaluate both styles by default
    ap.add_argument("--mode", type=str, default="both", choices=["both", "stateful", "stateless"])

    # Optional: set reward parameters for computing return
    ap.add_argument("--reward_mode", type=str, default="frozenlake_cost", choices=["frozenlake_cost", "shaped"])
    ap.add_argument("--step_cost", type=float, default=1.0)
    ap.add_argument("--goal_cost", type=float, default=0.0)
    ap.add_argument("--crash_cost", type=float, default=200.0)
    ap.add_argument("--action_cost", type=float, default=0.0)

    args = ap.parse_args()

    try:
        modes = ["stateful", "stateless"] if args.mode == "both" else [args.mode]
        for mode in modes:
            rep = run_episodes(
                episodes=args.episodes,
                seed=args.seed,
                mode=mode,
                reward_mode=args.reward_mode,
                step_cost=args.step_cost,
                goal_cost=args.goal_cost,
                crash_cost=args.crash_cost,
                action_cost=args.action_cost,
            )
            print("\n" + "=" * 80)
            print(f"REAP EVAL | mode={rep['mode']} | episodes={rep['episodes']} | seed={rep['seed']}")
            print(f"Avg ep len: {rep['avg_len_steps']:.2f} steps  ({rep['avg_len_seconds']:.2f} s)")
            print(f"Std ep len: {rep['std_len_steps']:.2f} steps")
            print(f"Avg return: {rep['avg_return']:.2f}   Std return: {rep['std_return']:.2f}")
            print("Outcomes:", rep["outcomes"])
            print("=" * 80)

    finally:
        # Close Isaac SimulationApp created inside go2_reap_ppo_mpc.py
        try:
            m.simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
