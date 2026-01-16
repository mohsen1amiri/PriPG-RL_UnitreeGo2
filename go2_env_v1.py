
class Go2REAPEnv(gym.Env):
    """
    Modified Environment to mimic the REAP MPC dynamics.

    - Observation: Robot World Position [x, y]
    - Action: Robot Body Velocity [v_x, v_y]
    - Physics: "Sliding" (Direct velocity control, ignoring legs)
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # --- Simulation timestep used by BOTH the env and (ideally) the planner ---
        self.DT_SIM = 0.01


        print(">> [Env] Creating World and loading Go2...")
        # Create world with desired dt (API may vary slightly across Isaac versions)
        self.world = World(backend="numpy")

        # # Try to force Isaac physics dt = DT_SIM
        # try:
        #     self.world.set_simulation_dt(physics_dt=self.DT_SIM, rendering_dt=self.DT_SIM)
        # except Exception:
        #     try:
        #         phys = self.world.get_physics_context()
        #         phys.set_time_step(self.DT_SIM)
        #     except Exception:
        #         print(">> [WARN] Could not set Isaac physics dt. Using Isaac default dt.")

        self.world.scene.add_default_ground_plane()

        assets_root = get_assets_root_path()
        go2_usd = assets_root + "/Isaac/Robots/Unitree/Go2/go2.usd"

        self.robot_prim_path = "/World/Go2"
        add_reference_to_stage(usd_path=go2_usd, prim_path=self.robot_prim_path)

        self.robot = SingleArticulation(prim_path=self.robot_prim_path, name="go2")
        self.world.scene.add(self.robot)

        # Expert planner (set from main). If None, no expert labels are produced.
        self.planner = None
        self.collect_expert = False   # ONLY True during imitation label collection



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


        # ----------------------------
        # Random start options
        # ----------------------------
        self.random_start = False
        self.start_clearance = 0.25
        self.start_border_clearance = 0.15
        self.start_max_tries = 200

        # ----------------------------
        # Crash handling option
        # ----------------------------
        self.terminate_on_crash = True  # default old behavior

    



    # ------------- Helpers -------------


    def get_expert_action(self, obs_xy: np.ndarray):
        """
        Returns expert action a*(s) for the given observation (state),
        without requiring env.step() to compute labels.
        """
        if self.planner is None:
            return None
        self.planner.reset()  # make expert depend ONLY on obs_xy
        return self.planner.get_action(obs_xy).copy()



    def _is_valid_start(self, xy: np.ndarray) -> bool:
        x, y = float(xy[0]), float(xy[1])

        # Stay away from borders
        m = float(self.robot_radius) + float(self.start_border_clearance)
        if not (self.x_min + m <= x <= self.x_max - m):
            return False
        if not (self.y_min + m <= y <= self.y_max - m):
            return False

        # Stay away from obstacles
        p = np.array([x, y], dtype=np.float32)
        for ox, oy, r in self.OBSTACLES:
            min_dist = float(r) + float(self.robot_radius) + float(self.start_clearance)
            if np.linalg.norm(p - np.array([ox, oy], dtype=np.float32)) <= min_dist:
                return False

        return True


    def _sample_start_xy(self) -> np.ndarray:
        # Uniform sampling inside arena with rejection
        m = float(self.robot_radius) + float(self.start_border_clearance)
        rng = getattr(self, "np_random", np.random)  # Gymnasium RNG if available
        for _ in range(int(self.start_max_tries)):
            x = rng.uniform(self.x_min + m, self.x_max - m)
            y = rng.uniform(self.y_min + m, self.y_max - m)
            xy = np.array([x, y], dtype=np.float32)
            if self._is_valid_start(xy):
                return xy


        # Fallback: original fixed start if sampling fails
        return np.array([-0.01, -1.5], dtype=np.float32)




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
        # traj_every <= 0 disables plots
        if self.traj_every <= 0:
            return

        # Save episodes: 1, 1+N, 1+2N, ...
        if ((self._episode_idx - 1) % self.traj_every) != 0:
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
        ax.scatter(traj[0, 0], traj[0, 1], marker="*", s=10, label="start") 
        ax.scatter(self.target_pos[0], self.target_pos[1], marker="o", s=10, label="goal")
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=10, label="end")



        # ---- Plot formatting ----
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.x_min - 0.2, self.x_max + 0.2)
        ax.set_ylim(self.y_min - 0.2, self.y_max + 0.2)
        ax.set_title(f"Episode {self._episode_idx} | event={event} | steps={traj.shape[0]-1}")
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
        # during-training evaluation
        p.add_argument("--eval_freq", type=int, default=0,
                    help="Run eval every N training timesteps (0 = disable)")
        p.add_argument("--eval_n_episodes", type=int, default=20,
                    help="How many episodes per eval during training")




        # ----------------------------
        # trajectory plots
        # ----------------------------
        p.add_argument("--traj_every_train", type=int, default=10,
                    help="Save trajectory PNG every N training episodes (0 disables).")
        p.add_argument("--traj_every_eval", type=int, default=10,
                    help="Save trajectory PNG every N eval episodes (0 disables).")
        p.add_argument("--save_eval_traj_during_train", action="store_true",
                    help="If set, periodic eval during training also saves trajectory PNGs.")


        # ----------------------------
        # random start
        # ----------------------------
        p.add_argument("--random_start", action="store_true",
                    help="Randomize start position (keeps away from obstacles/borders).")
        p.add_argument("--start_clearance", type=float, default=0.25,
                    help="Extra clearance (m) from obstacle+robot radius when sampling random starts.")
        p.add_argument("--start_border_clearance", type=float, default=0.15,
                    help="Extra clearance (m) from borders when sampling random starts.")
        p.add_argument("--start_max_tries", type=int, default=200,
                    help="Max rejection-sampling tries for random start.")

        # ----------------------------
        # crash handling
        # ----------------------------
        p.add_argument("--continue_on_crash", action="store_true",
                    help="If set, crashes do NOT end the episode (episode ends only on goal/timeout).")







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

        self.step_count = 0

        self.world.reset()
        self.robot.initialize()

        # Start position
        if self.random_start:
            xy = self._sample_start_xy()
            self.robot.set_world_pose(position=np.array([float(xy[0]), float(xy[1]), 0.35]))
        else:
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
        if self.collect_expert and (self.planner is not None):
            self.planner.reset()
            expert_u = self.planner.get_action(obs_t).copy()


        # Save pose before stepping (for crash rollback)
        pos_before, orient_before = self.robot.get_world_pose()


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
        timeout_now = (self.step_count >= self.max_steps)

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
            info["is_success"] = 1.0
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

            # pick event label
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
                "border_side": border_side,
            }
            info["is_success"] = 0.0
            if expert_u is not None:
                info["Action MPC"] = expert_u

            if self.terminate_on_crash:
                terminated = True
                truncated = False
                self._save_episode_trajectory(event=event)
                return obs, reward, terminated, truncated, info

            # ----------------------------
            # NEW: continue-on-crash mode
            # rollback to last safe pose and keep going
            # ----------------------------
            # Remove the crashed point we already appended this step
            # (so plots do not show an invalid point inside obstacle/wall)
            if len(self._traj_xy) > 0:
                self._traj_xy.pop()

            self.robot.set_world_pose(position=pos_before, orientation=orient_before)
            self.robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            self.robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

            # optional: one settle step (keeps minimal & stable)
            self.world.step(render=False)

            # Refresh observation after rollback
            obs = self._get_obs()
            self._traj_xy.append(obs.copy())
            
            dist = float(np.linalg.norm(obs - self.target_pos))
            info["dist"] = dist

            # keep shaped progress consistent after rollback
            if self.reward_mode == "shaped":
                self.prev_dist = float(dist)



            # Mark crash happened, but do NOT end episode
            info["crash_continue"] = True

            # IMPORTANT: enforce timeout even when continuing after crash
            if timeout_now:
                info["event"] = "timeout"
                info["is_success"] = 0.0
                self._save_episode_trajectory(event="timeout")
                return obs, reward, False, True, info

            return obs, reward, False, False, info






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
            info["is_success"] = 0.0 
            self._save_episode_trajectory(event=event)
            return obs, reward, False, True, info

        return obs, reward, terminated, truncated, info