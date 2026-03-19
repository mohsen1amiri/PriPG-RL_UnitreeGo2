import torch
import time
from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext


# -----------------------------
# 1. Launch Isaac Lab
# -----------------------------
app_launcher = AppLauncher(headless=False)   # Set True for Alvis batch mode
app = app_launcher.app


# -----------------------------
# 2. Define a minimal environment
# -----------------------------
class Go2VelocityTestEnv(DirectRLEnv):
    def __init__(self):
        super().__init__(use_viewer=True)

        # Load simulation
        self.sim = SimulationContext()

        # Load Unitree Go2
        self.robot = Articulation(
            prim_path="/World/Go2",
            usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/2023.1/Robots/Unitree/Go2/go2.usd",
            name="go2",
            translation=torch.tensor([0.0, 0.0, 0.4])
        )
        self.robot.initialize()

        # Command buffer: (1 robot, 3 command dimensions)
        self.commands = torch.zeros((1, 3), dtype=torch.float32)

    def pre_physics_step(self, actions):
        """Apply commands to the locomotion controller."""
        self.commands[:] = actions
        # Isaac Lab internal controller will read this tensor automatically

    def get_observations(self):
        """Observations not needed, but required by API."""
        return {"obs": torch.zeros((1, 1))}

    def get_rewards(self):
        """No rewards needed in this test."""
        return torch.zeros((1,))

    def get_dones(self):
        return torch.zeros((1,), dtype=torch.bool)


# -----------------------------
# 3. Run the environment
# -----------------------------
def main():
    env = Go2VelocityTestEnv()

    print("\n==============================")
    print(" Unitree Go2 Velocity Test")
    print("==============================\n")

    # Commands to test
    test_commands = [
        (0.5, 0.0, 0.0),   # forward
        (0.0, 0.5, 0.0),   # sideways
        (0.0, 0.0, 1.0),   # rotate
        (0.5, 0.5, 0.5),   # mix
        (0.0, 0.0, 0.0)    # stop
    ]

    step_hz = 200
    dt = 1.0 / step_hz

    for cmd in test_commands:
        vx, vy, yaw = cmd
        print(f"\nSending command: vx={vx}, vy={vy}, yaw={yaw}")

        start_time = time.time()

        # Run each command for 2 seconds
        while time.time() - start_time < 2.0:

            # Step simulation
            env.sim.step()
            env.pre_physics_step(torch.tensor([[vx, vy, yaw]]))

            # Read true rigid body velocity
            base_vel = env.robot.root_state_w[:, 7:10][0]

            print(f"  Commanded: {cmd} | Measured linear = {base_vel.numpy()}", end="\r")

            time.sleep(dt)

    print("\n\nFinished test!")


if __name__ == "__main__":
    main()