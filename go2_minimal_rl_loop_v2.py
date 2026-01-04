"""
Modified Minimal PPO for Unitree Go2 in Isaac Sim
Behaving like a 2D Point Mass (REAP Style)
"""

# -------- Start Isaac Sim headless ----------
from isaacsim.simulation_app import SimulationApp
# Toggle headless=False if you want to watch it slide!
simulation_app = SimulationApp({"headless": True})

# -------- Standard imports ----------
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sympy as sp

from stable_baselines3 import PPO

# -------- Isaac Sim Core API imports ----------
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path



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
        self.target = np.array([-0.1, -1.0])

        # Obstacles (x, y, radius)
        self.OBSTACLES = np.array([
            [0, 1.5, .25],  
            [.0, 0.0, 0.25],
            [-.750, 0.750, 0.25],
            [-.750, -0.750, 0.25],
            [.750, 0.750, 0.25],
            [.750, -0.750, 0.25]
        ])

        # --- SYMBOLIC MATH SETUP ---
        # (This is copied from your code and wrapped in the class)
        x_sym = sp.Matrix([sp.Symbol('x'), sp.Symbol('y')])
        u_sym = sp.Matrix([sp.Symbol('u_x'), sp.Symbol('u_y')])
        
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
        self.num_constraints_step = sp.lambdify(list(u_sym) + list(x_sym), all_constraints_sym, 'numpy')
        dCdu_sym = [c_i.diff(u_var) for c_i in all_constraints_sym for u_var in u_sym]
        self.num_dCdu_step = sp.lambdify(list(u_sym) + list(x_sym), dCdu_sym, 'numpy')
        dCdx_sym = [c_i.diff(x_var) for c_i in all_constraints_sym for x_var in x_sym]
        self.num_dCdx_step = sp.lambdify(list(u_sym) + list(x_sym), dCdx_sym, 'numpy')

        # --- INTERNAL STATE (Warm Start) ---
        self.opt_vars = np.zeros(self.N_HORIZON * 2) 
        self.hat_lambda = np.zeros(self.N_HORIZON * self.c_per_step)

    def get_action(self, current_pos):
        """
        Takes current Robot Position [x, y]
        Returns Optimal Velocity [vx, vy]
        """
        # Run a few gradient steps per simulation frame to converge better
        # (Your original code did 1 step, we do 5 here for robustness)
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
            self.hat_lambda = np.maximum(self.hat_lambda, 0) # Project >= 0
            self.opt_vars = np.clip(self.opt_vars, -self.MAX_VEL, self.MAX_VEL)

        # Extract first action (Model Predictive Control)
        u_applied = self.opt_vars[:2].copy()

        # Shift Warm Start (Prepare for next frame)
        self.opt_vars = np.roll(self.opt_vars, -2)
        self.opt_vars[-2:] = 0.0
        
        return u_applied

    # --- Internal Helpers (Simplified from your code) ---
    def _get_trajectory(self, u_flat, x0):
        U = u_flat.reshape(self.N_HORIZON, 2)
        X = np.zeros((self.N_HORIZON + 1, 2))
        X[0] = x0
        for k in range(self.N_HORIZON):
            X[k+1] = X[k] + U[k] * self.DT_SIM
        return U, X

    def _compute_grad_u(self, u_flat, x0, lambdas_flat):
        U, X = self._get_trajectory(u_flat, x0)
        lambdas = lambdas_flat.reshape(self.N_HORIZON, self.c_per_step)
        grad_U = np.zeros((self.N_HORIZON, 2))
        grad_x_accum = np.zeros(2) 

        for k in reversed(range(self.N_HORIZON)):
            x_curr, x_next, u_curr = X[k], X[k+1], U[k]
            
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
                if log_arg < 1e-6: log_arg = 1e-6 
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

        # Reset to load physics handles
        self.world.reset()
        self.robot.initialize()

        # --- 1. DEFINE TARGET (From your REAP code) ---
        self.target_pos = np.array([-0.1, -1.0], dtype=np.float32)

        # --- 2. MODIFY ACTION SPACE (Velocity vx, vy) ---
        # Bounds: -2.0 to 2.0 m/s (Matches your MAX_LINEAR_VEL)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)

        # --- 3. MODIFY OBSERVATION SPACE (Position x, y) ---
        # Bounds: Large enough to cover the room
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(2,), dtype=np.float32)

        print(f">> [Env] Action Space: {self.action_space}")
        print(f">> [Env] Observation Space: {self.observation_space}")

    # ------------- Helpers -------------

    def _get_obs(self) -> np.ndarray:
        # Get global position from Isaac Sim
        pos, orient = self.robot.get_world_pose()
        
        # Return only X and Y
        obs = np.array([pos[0], pos[1]], dtype=np.float32)
        return obs

    # ------------- Gym API -------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print(">> [Env] reset() called")

        self.world.reset()
        self.robot.initialize()

        # Set Start Position (From your REAP code: [-0.01, 3.0])
        # We set Z=0.35 so it doesn't spawn inside the floor
        self.robot.set_world_pose(position=np.array([-0.01, 3.0, 0.35]))
        
        # Stop any movement
        self.robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self.robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

        # Settle
        for _ in range(10):
            self.world.step(render=False)

        return self._get_obs(), {}

    def step(self, action):
        # Action is [vx, vy]
        # Clip to ensure safety
        vx, vy = np.clip(action, -2.0, 2.0)

        # --- THE MAGIC TRICK ---
        # Instead of moving joints, we force the body to slide.
        # We set Z velocity to 0.0 to keep it on the ground plane.
        velocity_command = np.array([vx, vy, 0.0])
        self.robot.set_linear_velocity(velocity_command)

        # Step Physics
        self.world.step(render=False)

        # Get Observation
        obs = self._get_obs()

        # Calculate Reward
        # Reward is negative distance to target (Closer = Better)
        dist_to_target = np.linalg.norm(obs - self.target_pos)
        reward = -dist_to_target

        # Check termination (Goal Reached)
        terminated = False
        if dist_to_target < 0.1: # POSITION_TOLERANCE from REAP
            reward += 100.0 # Bonus for reaching goal
            terminated = True
            print(">> [Env] Target Reached!")

        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info


def main():
    print(">> [Main] Creating Go2MPCEnv (Sliding Mode)...")
    env = Go2MPCEnv()
    print(">> [Main] Initialize REAP Brain...")
    # 2. Initialize REAP Brain
    planner = REAP_Planner()

    print(">> [Main] Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-3, # Higher LR because this task is easier
        n_steps=2048,
        batch_size=64,
    )

    print(">> [Main] Starting PPO training...")
    # It should learn very quickly because sliding is easier than walking
    model.learn(total_timesteps=10000)
    
    print(">> [Main] Training finished, saving model...")
    model.save("ppo_go2_sliding_mpc_style")
    print(">> [Main] Done.")

    print(">> [Main] Evaluating REAP controller in env...")
    obs, _ = env.reset()

    for t in range(1000):
        current_pos = obs                  # obs is [x, y]
        action_vel = planner.get_action(current_pos)
        obs, reward, terminated, truncated, info = env.step(action_vel)

        dist = np.linalg.norm(current_pos - planner.target)
        if dist < 0.1 or terminated or truncated:
            print(f">> GOAL REACHED at step {t}, dist={dist:.3f}")
            break



if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()