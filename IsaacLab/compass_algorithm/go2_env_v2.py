import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# --- CONFIG IMPORTS ---
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

from isaaclab.managers import SceneEntityCfg

# =====================================================================
# 1. SCENE CONFIGURATION
# =====================================================================
@configclass
class ReapGo2EnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- SYNC TIMING ---
        # Physics step (dt) is 0.005s. Decimation of 4 means 
        # Env step = 0.005 * 4 = 0.02s.
        self.sim.dt = 0.005 
        self.decimation = 4

        # 1. CREATE OBSTACLES (Exact match to REAP Planner)
        obstacles = [
            (0.0,   0.15, 0.23), (-1.3,  0.75, 0.23), (0.0,   1.45, 0.23),
            (1.3,   0.75, 0.23), (1.3,  -0.45, 0.23), (-1.3, -0.45, 0.23)
        ]
        for i, (ox, oy, r) in enumerate(obstacles):
            setattr(self.scene, f"reap_obs_{i}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/reap_obs_{i}",
                spawn=sim_utils.CylinderCfg(
                    func=sim_utils.spawn_cylinder,
                    radius=float(r), height=1.0, 
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(float(ox), float(oy), 0.5)),
            ))

        # 2. CREATE GOAL (Exact match to REAP Planner Target)
        self.scene.reap_goal = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Goal",
            spawn=sim_utils.SphereCfg(
                func=sim_utils.spawn_sphere,
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 2.8, 0.1)),
        )

        # 3. CREATE BORDER WALLS
        wall_thickness, wall_height = 0.1, 1.0
        x_len, y_len = 4.1, 5.6 
    
        walls = {
            "border_bottom": ((x_len, wall_thickness, wall_height), (0.0, -2.05, 0.5)),
            "border_top":    ((x_len, wall_thickness, wall_height), (0.0, 3.55, 0.5)),
            "border_left":   ((wall_thickness, y_len, wall_height), (-2.05, 0.75, 0.5)),
            "border_right":  ((wall_thickness, y_len, wall_height), (2.05, 0.75, 0.5)),
        }
        for name, (scale, pos) in walls.items():
            setattr(self.scene, name, AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=sim_utils.CuboidCfg(
                    func=sim_utils.spawn_cuboid,
                    size=tuple(float(s) for s in scale),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 1.0)),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(float(p) for p in pos)),
            ))

        # --- VIEWER SETTINGS ---
        if self.viewer is not None:
            self.viewer.resolution = (1280, 720)
            self.viewer.eye = (0.0, 0.75, 10.0) 
            self.viewer.lookat = (0.0, 0.75, 0.0)
        
        # --- TERMINATIONS ---
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        import isaaclab.envs.mdp as mdp

        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "threshold": 1.0,  
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), 
            }
        )

# =====================================================================
# 2. ENVIRONMENT CLASS
# =====================================================================
class Go2IsaacLabNavEnv_v2(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
        planner=None,
        num_envs=1,
        device="cuda:0",
        render_mode=None,
        max_steps=6000,
        goal_reward=100.0,
        crash_cost=200.0,
        step_cost=1.0,
        w_progress=50.0,
        robot_radius=0.3,
        leg_reach_margin=0.02,
        wall_margin=0.35,
        goal_tol=0.05,
        action_mode="x_yaw",
        force_expert_control=False
        ):
        super().__init__()



        self.action_mode = action_mode
        if self.action_mode not in ["xy", "xyyaw", "x_yaw"]:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

        self.planner = planner

        self.num_envs = num_envs

        self.device = device

        self.render_mode = render_mode

        self.force_expert_control = force_expert_control
        
        if self.force_expert_control and self.planner is None:
            raise ValueError("force_expert_control=True requires planner to be provided.")


        self.target_pos = np.array([0.0, 2.8], dtype=np.float32)


        self.M_t = 0.0

        self.total_t = 0

        self.goal_tol = goal_tol

        self.max_steps = max_steps

        self.step_count = 0

        self.step_cost = step_cost

        self.goal_reward = goal_reward

        self.crash_cost = crash_cost

        self.w_progress = w_progress

        self.prev_dist = 0.0



        self.robot_radius = robot_radius

        self.leg_reach_margin = leg_reach_margin

        self.wall_margin = wall_margin

        self.OBSTACLES = np.array([
            [0.0,   0.15, 0.23], [-1.3,  0.75, 0.23], [0.0,   1.45, 0.23],
            [1.3,   0.75, 0.23], [1.3,  -0.45, 0.23], [-1.3, -0.45, 0.23],
        ], dtype=np.float32)
        
        self.x_min, self.x_max = -2.0, 2.0
        self.y_min, self.y_max = -2.0, 3.5

        env_cfg = ReapGo2EnvCfg()
        env_cfg.scene.num_envs = self.num_envs
        env_cfg.scene.env_spacing = 5.0
        self.task_name = "Isaac-Velocity-Flat-Unitree-Go2-v0"
        self.base_env = gym.make(self.task_name, cfg=env_cfg, render_mode=self.render_mode)
        
        # Rendering Sync
        if self.render_mode == "rgb_array":
            print(">> [Env] Priming rendering pipeline...")
            self.base_env.reset()
            import omni.kit.app
            app = omni.kit.app.get_app()
            success = False
            attempts = 0
            while not success and attempts < 50:
                try:
                    app.update()
                    self.base_env.unwrapped.sim.render()
                    self.base_env.render()
                    success = True
                except TypeError:
                    attempts += 1
            if not success:
                raise RuntimeError("Renderer failed to initialize.")

        print(">> [Env] Loading Pretrained Locomotion Policy...")
        vec_env = RslRlVecEnvWrapper(self.base_env)
        agent_cfg = load_cfg_from_registry(self.task_name, "rsl_rl_cfg_entry_point")
        self.runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        
        ckpt = ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
        self.runner.load(ckpt)
        self.locomotion_policy = self.runner.get_inference_policy(device=self.device)

         
        if self.action_mode == "xyyaw":
            self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32)
        elif self.action_mode in ["xy", "x_yaw"]:
            self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")

        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(4,), dtype=np.float32)

        self.ep_path_length = 0.0
        self.ep_energy = 0.0
        self.start_pos = None
        self.last_pos = None

    def _get_obs(self, robot_xy):
        return np.array([robot_xy[0], robot_xy[1], self.target_pos[0], self.target_pos[1]], dtype=np.float32)

    def set_compass_status(self, is_mature: float, total_steps: int):
        self.M_t = is_mature
        self.total_t = total_steps

    def render(self):
        if self.render_mode == "rgb_array":
            return self.base_env.render()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        if self.planner is not None:
            self.planner.reset()
    
        self.current_obs_dict, _ = self.base_env.reset() 

        # Rejection Sampling for Spawn
        spawn_z = 0.5
        valid_spawn = False
        while not valid_spawn:
            test_x = self.np_random.uniform(self.x_min + self.wall_margin, self.x_max - self.wall_margin)
            test_y = self.np_random.uniform(self.y_min + self.wall_margin, 0.0)
            is_safe = True
            for ox, oy, r in self.OBSTACLES:
                dist_to_obs = np.linalg.norm([test_x - ox, test_y - oy])
                if dist_to_obs <= (r + self.robot_radius + self.leg_reach_margin):
                    is_safe = False
                    break
            if is_safe:
                spawn_x, spawn_y = test_x, test_y
                valid_spawn = True
        
        new_pos = torch.tensor([[spawn_x, spawn_y, spawn_z]], device=self.device)
        new_rot = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]], device=self.device)
        self.base_env.unwrapped.scene["robot"].write_root_pose_to_sim(torch.cat([new_pos, new_rot], dim=-1))
        self.base_env.unwrapped.sim.step() 
        
        robot_xy = np.array([spawn_x, spawn_y], dtype=np.float32)
        self.prev_dist = float(np.linalg.norm(robot_xy - self.target_pos))
        self.start_pos = np.copy(robot_xy)
        self.last_pos = np.copy(robot_xy)
        self.ep_path_length = 0.0
        self.ep_energy = 0.0
        
        return self._get_obs(robot_xy), {}

    def step(self, action):
        self.step_count += 1
        act_tensor = torch.tensor(action, device=self.device, dtype=torch.float32)
        
        root_pos_t = self.base_env.unwrapped.scene["robot"].data.root_pos_w
        robot_xy_t = root_pos_t[0, :2].cpu().numpy()
        root_quat_t = self.base_env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()
        w, x, y, z = root_quat_t[0], root_quat_t[1], root_quat_t[2], root_quat_t[3]
        robot_yaw_t = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        expert_action_array = None

        if self.planner is not None:
            expert_global = self.planner.get_action(robot_xy_t)
            u_xw, u_yw = expert_global[0], expert_global[1]

            # common heading quantities
            V = np.hypot(u_xw, u_yw)
            target_yaw = np.arctan2(u_yw, u_xw)
            e_t = target_yaw - robot_yaw_t
            e_t = (e_t + np.pi) % (2 * np.pi) - np.pi

            if self.action_mode == "xyyaw":
                # 3D: (vx, vy, yaw_rate)
                L, rho = 0.1, 1.0 # L, rho = 0.3, 0.99
                exp_vx = V * np.cos(e_t)
                exp_vy = (1.0 - rho) * V * np.sin(e_t)
                exp_yaw = rho * (V * np.sin(e_t)) / L

                expert_action_array = np.array([
                    np.clip(exp_vx, -2.0, 2.0),
                    np.clip(exp_vy, -2.0, 2.0),
                    np.clip(exp_yaw, -2.0, 2.0)
                ], dtype=np.float32)

            elif self.action_mode == "xy":
                # 2D: (vx, vy)   [your current 2D mode]
                expert_action_array = np.array([
                    np.clip(u_yw, -2.0, 2.0),
                    np.clip(-u_xw, -2.0, 2.0)
                ], dtype=np.float32)

            elif self.action_mode == "x_yaw":
                # 2D: (vx, yaw_rate)
                L = 0.3
                exp_vx = V * np.cos(e_t)
                exp_yaw = (V * np.sin(e_t)) / L

                expert_action_array = np.array([
                    np.clip(exp_vx, -2.0, 2.0),
                    np.clip(exp_yaw, -2.0, 2.0)
                ], dtype=np.float32)

            else:
                raise ValueError(f"Unsupported action_mode: {self.action_mode}")

        # Default behavior: execute policy action
        executed_action_tensor = act_tensor

        # Override only in pure expert mode
        if self.force_expert_control:
            if self.planner is None:
                raise RuntimeError("force_expert_control=True but planner is None.")
            if expert_action_array is None:
                raise RuntimeError("Expert control requested but planner did not provide an action.")
            executed_action_tensor = torch.tensor(
                expert_action_array, device=self.device, dtype=torch.float32
            )

        if self.action_mode == "xyyaw":
            self.current_obs_dict["policy"][:, 9] = executed_action_tensor[0]   # vx
            self.current_obs_dict["policy"][:, 10] = executed_action_tensor[1]  # vy
            self.current_obs_dict["policy"][:, 11] = executed_action_tensor[2]  # yaw_rate

        elif self.action_mode == "xy":
            self.current_obs_dict["policy"][:, 9] = executed_action_tensor[0]   # vx
            self.current_obs_dict["policy"][:, 10] = executed_action_tensor[1]  # vy
            self.current_obs_dict["policy"][:, 11] = 0.0                        # yaw_rate

        elif self.action_mode == "x_yaw":
            self.current_obs_dict["policy"][:, 9] = executed_action_tensor[0]   # vx
            self.current_obs_dict["policy"][:, 10] = 0.0                        # vy
            self.current_obs_dict["policy"][:, 11] = executed_action_tensor[1]  # yaw_rate

        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")

        vel_command = self.base_env.unwrapped.command_manager.get_command("base_velocity")
        if self.action_mode == "xyyaw":
            vel_command[:, 0] = executed_action_tensor[0]  # vx
            vel_command[:, 1] = executed_action_tensor[1]  # vy
            vel_command[:, 2] = executed_action_tensor[2]  # yaw_rate

        elif self.action_mode == "xy":
            vel_command[:, 0] = executed_action_tensor[0]  # vx
            vel_command[:, 1] = executed_action_tensor[1]  # vy
            vel_command[:, 2] = 0.0

        elif self.action_mode == "x_yaw":
            vel_command[:, 0] = executed_action_tensor[0]  # vx
            vel_command[:, 1] = 0.0
            vel_command[:, 2] = executed_action_tensor[1]  # yaw_rate

        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")

        with torch.inference_mode():
            ll_actions = self.locomotion_policy(self.current_obs_dict)

        self.current_obs_dict, _, base_terminated, _, _ = self.base_env.step(ll_actions)

        # Accumulate metrics
        root_pos_t1_metrics = self.base_env.unwrapped.scene["robot"].data.root_pos_w[0, :2].cpu().numpy()
        self.ep_path_length += float(np.linalg.norm(root_pos_t1_metrics - self.last_pos))
        self.last_pos = np.copy(root_pos_t1_metrics)
        self.ep_energy += torch.sum(torch.square(ll_actions[0])).item()

        # Navigation logic
        root_pos_t1 = self.base_env.unwrapped.scene["robot"].data.root_pos_w
        robot_xy_t1 = root_pos_t1[0, :2].cpu().numpy()
        robot_z_t1 = root_pos_t1[0, 2].item()

        reward, terminated, truncated, info = self._compute_navigation_logic(robot_xy_t1)

        if expert_action_array is not None:
            info["expert_action"] = expert_action_array

        physically_crashed = base_terminated.any().item() if hasattr(base_terminated, "any") else bool(base_terminated)
        is_fallen = bool(robot_z_t1 < 0.1)

        if physically_crashed or is_fallen:
            terminated = True
            reward = -self.crash_cost
            info["is_success"] = 0.0

        # Final episode metrics
        if terminated or truncated:
            info["is_crash"] = 1.0 if (physically_crashed or is_fallen or (info.get("is_success", 0.0) == 0.0 and not truncated)) else 0.0
            if info.get("is_success", 0.0) == 1.0:
                direct_dist = float(np.linalg.norm(self.target_pos - self.start_pos))
                runtime_seconds = self.step_count * self.base_env.unwrapped.step_dt

                info["success_path_optimality"] = self.ep_path_length / direct_dist if direct_dist > 0.0 else 1.0
                info["success_energy"] = self.ep_energy
                info["success_runtime"] = runtime_seconds
                info["success_velocity"] = self.ep_path_length / runtime_seconds if runtime_seconds > 0.0 else 0.0

        return self._get_obs(robot_xy_t1), reward, terminated, truncated, info


    def _compute_navigation_logic(self, robot_xy):
        dist = np.linalg.norm(robot_xy - self.target_pos)
        progress = self.prev_dist - dist
        self.prev_dist = float(dist)
        info = {"dist": float(dist), "progress": float(progress)}
        
        if dist < self.goal_tol:
            info["is_success"] = 1.0 
            return self.goal_reward, True, False, info

        for ox, oy, r in self.OBSTACLES:
            if np.linalg.norm(robot_xy - np.array([ox, oy])) <= (r + self.robot_radius + self.leg_reach_margin):
                info["is_success"] = 0.0 
                return -self.crash_cost, True, False, info
                
        x, y = robot_xy[0], robot_xy[1]
        if (x < self.x_min + self.wall_margin) or (x > self.x_max - self.wall_margin) or \
           (y < self.y_min + self.wall_margin) or (y > self.y_max - self.wall_margin):
            info["is_success"] = 0.0 
            return -self.crash_cost, True, False, info
            
        if self.step_count >= self.max_steps:
            info["is_success"] = 0.0 
            return -self.step_cost, False, True, info
            
        reward = (self.w_progress * progress) - self.step_cost
        return reward, False, False, info