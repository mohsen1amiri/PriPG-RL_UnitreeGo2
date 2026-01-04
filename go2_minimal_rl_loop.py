"""
Minimal PPO + Stable-Baselines3 training loop for Unitree Go2 in Isaac Sim
(using isaacsim.core.api directly, no Isaac Lab)

Run:
    isaacrun go2_minimal_rl_loop.py
"""

# -------- Start Isaac Sim headless ----------
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": True})

# -------- Standard imports ----------
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO

# -------- Isaac Sim Core API imports ----------
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.types import ArticulationAction


class Go2IsaacEnv(gym.Env):
    """
    Very simple Gymnasium environment around Unitree Go2 in Isaac Sim.

    - Observation: joint positions (all DOFs) as a 1D float32 vector.
    - Action: joint position deltas in radians, clipped to [-0.5, 0.5].
    - Reward: negative L2 distance from default joint configuration.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        print(">> [Env] Creating World and loading Go2...")
        self.world = World(backend="numpy")  # dt ~ 1/60 s
        self.world.scene.add_default_ground_plane()

        assets_root = get_assets_root_path()
        go2_usd = assets_root + "/Isaac/Robots/Unitree/Go2/go2.usd"

        self.robot_prim_path = "/World/Go2"
        add_reference_to_stage(usd_path=go2_usd, prim_path=self.robot_prim_path)

        self.robot = SingleArticulation(prim_path=self.robot_prim_path, name="go2")
        self.world.scene.add(self.robot)

        # First reset to initialize physics + articulation
        self.world.reset()
        self.robot.initialize()

        q0 = self.robot.get_joint_positions()
        q0 = np.array(q0, dtype=np.float32)
        self.n_dof = q0.shape[0]
        print(f">> [Env] n_dof = {self.n_dof}")

        self.default_q = q0.copy()
        self.controller = self.robot.get_articulation_controller()

        # ----- Gym spaces -----
        act_low = -0.5 * np.ones(self.n_dof, dtype=np.float32)
        act_high = 0.5 * np.ones(self.n_dof, dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        obs_low = -np.pi * np.ones(self.n_dof, dtype=np.float32)
        obs_high = np.pi * np.ones(self.n_dof, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    # ------------- Helpers -------------

    def _get_obs(self) -> np.ndarray:
        q = self.robot.get_joint_positions()
        q = np.array(q, dtype=np.float32)
        return q

    def _apply_joint_targets(self, target_q: np.ndarray):
        action = ArticulationAction(joint_positions=target_q)
        self.controller.apply_action(action)

    # ------------- Gym API -------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print(">> [Env] reset() called")

        # Reset physics world
        self.world.reset()
        self.robot.initialize()

        # Start from default pose
        self._apply_joint_targets(self.default_q)

        # Let it settle a bit
        for i in range(10):
            print(f">> [Env] settle step {i}")
            self.world.step(render=False)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

        current_q = self._get_obs()
        target_q = current_q + action
        self._apply_joint_targets(target_q)

        self.world.step(render=False)

        obs = self._get_obs()
        pose_error = np.linalg.norm(obs - self.default_q)
        reward = -pose_error

        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info


def main():

    print(">> [Main] Creating single Go2IsaacEnv (no VecEnv)...")
    env = Go2IsaacEnv()

    # Optional sanity reset
    print(">> [Main] Doing one reset before training...")
    obs, info = env.reset()
    print(">> [Main] First obs shape:", obs.shape)

    print(">> [Main] Creating PPO model...")
    model = PPO(
            policy="MlpPolicy",
            env=env,          # pass env directly, SB3 will wrap it internally
            verbose=1,
            n_steps=512,
            batch_size=64,
            gae_lambda=0.95,
            gamma=0.99,
            n_epochs=10,
            learning_rate=3e-4,
            clip_range=0.2,
        )

    print(">> [Main] Starting PPO training...")
    model.learn(total_timesteps=2000)  # increase later
    print(">> [Main] Training finished, saving model...")
    model.save("ppo_go2_isaacsim_sb3_single_env")
    print(">> [Main] Done.")
    


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(">> [PYTHON ERROR] Unhandled exception in main():", repr(e))
        traceback.print_exc()
    finally:
        simulation_app.close()

