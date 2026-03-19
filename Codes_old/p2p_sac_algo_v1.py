import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update



def _linear_schedule(t: int, start: float, end: float, end_steps: int) -> float:
    if end_steps <= 0:
        return float(end)
    frac = min(1.0, float(t) / float(end_steps))
    return float(start + frac * (end - start))


class P2PReplayBuffer(ReplayBuffer):
    """
    ReplayBuffer that ALSO stores:
      - expert_action (a*)
      - has_expert (mask)
    Assumes n_envs == 1 (DummyVecEnv). That matches your current setup.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if getattr(self, "n_envs", 1) != 1:
            raise ValueError("P2PReplayBuffer currently supports n_envs == 1 only.")

        # SB3 stores actions as (buffer_size, n_envs, action_dim)
        self.expert_actions = np.zeros_like(self.actions, dtype=np.float32)

        # mask must match (buffer_size, n_envs)
        self.has_expert = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)


    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        # store expert label for THIS transition position
        pos = self.pos  # position BEFORE super().add increments it

        info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
        a_star = info0.get("expert_action", None)
        has = bool(info0.get("has_expert", False)) and (a_star is not None)

        if has:
            self.expert_actions[pos, 0, :] = np.asarray(a_star, dtype=np.float32)
            self.has_expert[pos, 0] = 1.0
        else:
            self.has_expert[pos, 0] = 0.0


        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env=None, return_indices: bool = False):
        # Re-implement the simple SB3 sampling (works for n_envs == 1).
        upper_bound = self.buffer_size if self.full else self.pos
        rng = getattr(self, "rng", None)
        if rng is not None and hasattr(rng, "integers"):
            batch_inds = rng.integers(0, upper_bound, size=batch_size)
        else:
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)


        data = self._get_samples(batch_inds, env=env)
        if return_indices:
            return data, batch_inds
        return data


class P2P_SAC(SAC):
    """
    SAC + P2P:
      - Behavior: mix expert with policy (prob p_expert(t)), plus expert_noise.
      - Labels: query expert with probability label_prob(t) even if we execute policy.
      - Update: after normal SAC train(), do an extra Q-filtered BC actor step.
    """

    def __init__(
        self,
        *args,
        prefill_steps: int = 20_000,
        p_expert0: float = 1.0,
        p_expert_end: float = 0.0,
        p_expert_end_steps: int = 200_000,
        label_prob: float = 1.0,          # probability to query a* when executing policy
        expert_noise: float = 0.1,        # Gaussian noise added to expert action when executed
        bc_lambda0: float = 1.0,
        bc_lambda_end: float = 0.0,
        bc_lambda_end_steps: int = 200_000,
        qfilter_warmup_steps: int = 0,
        qfilter_tau: float = 1.0,         # sigmoid temperature
        bc_gradient_steps: int = 1,        # extra BC updates per SAC train() step
        **kwargs,
    ):
        self.prefill_steps = int(prefill_steps)
        self.qfilter_warmup_steps = int(qfilter_warmup_steps)
        self.p_expert0 = float(p_expert0)
        self.p_expert_end = float(p_expert_end)
        self.p_expert_end_steps = int(p_expert_end_steps)

        self.label_prob = float(label_prob)
        self.expert_noise = float(expert_noise)

        self.bc_lambda0 = float(bc_lambda0)
        self.bc_lambda_end = float(bc_lambda_end)
        self.bc_lambda_end_steps = int(bc_lambda_end_steps)

        self.qfilter_tau = float(qfilter_tau)
        self.bc_gradient_steps = int(bc_gradient_steps)

        self._last_expert_action = None
        self._last_has_expert = False
        self._last_expert_action_scaled = None


        super().__init__(*args, **kwargs)

    # ---------- expert access ----------

    def _unwrap_env0(self):
        # DummyVecEnv -> envs[0] -> possibly Monitor -> .env -> actual env
        env0 = self.get_env().envs[0]
        base = env0
        for _ in range(10):
            if hasattr(base, "env"):
                base = base.env
                continue
            if hasattr(base, "unwrapped"):
                base = base.unwrapped
            break
        return base

    def _query_expert(self, obs_np: np.ndarray):
        base = self._unwrap_env0()
        if not hasattr(base, "get_expert_action"):
            return None
        return base.get_expert_action(obs_np)

    # ---------- schedules ----------

    def _p_expert(self) -> float:
        t = int(self.num_timesteps)
        return _linear_schedule(t, self.p_expert0, self.p_expert_end, self.p_expert_end_steps)

    def _bc_lambda(self) -> float:
        t = int(self.num_timesteps)
        return _linear_schedule(t, self.bc_lambda0, self.bc_lambda_end, self.bc_lambda_end_steps)

    # ---------- behavior policy mixing ----------

    def _rand01(self) -> float:
        rng = getattr(self, "np_random", None)
        if rng is None:
            return float(np.random.rand())
        # Gymnasium often gives Generator-like RNG
        if hasattr(rng, "random"):
            return float(rng.random())
        # fallback
        return float(rng.rand())


    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1):
        # get the default SAC action first (handles scaling)
        action, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)

        # we only support n_envs==1 here
        obs = self._last_obs
        obs0 = obs[0] if obs.ndim > 1 else obs

        # Decide whether to use expert action this step
        use_expert = False
        if int(self.num_timesteps) < self.prefill_steps:
            use_expert = True
        else:
            if self._rand01() < self._p_expert():
                use_expert = True

        self._last_expert_action = None
        self._last_expert_action_scaled = None
        self._last_has_expert = False


        if use_expert:
            a_star = self._query_expert(obs0)
            if a_star is not None:
                a_exec = np.asarray(a_star, dtype=np.float32).copy()
                if self.expert_noise > 0:
                    a_exec = a_exec + np.random.randn(*a_exec.shape).astype(np.float32) * self.expert_noise

                # clip to env bounds
                low, high = self.action_space.low, self.action_space.high
                a_exec = np.clip(a_exec, low, high)

                action = a_exec[None, :]  # (1, action_dim)
                buffer_action = self.policy.scale_action(action)

                a_star_env = np.asarray(a_star, dtype=np.float32)
                a_star_scaled = self.policy.scale_action(a_star_env[None, :])[0]

                self._last_expert_action = a_star_env
                self._last_expert_action_scaled = a_star_scaled
                self._last_has_expert = True

        else:
            # we executed policy; optionally still query expert label for training
            if self._rand01() < self.label_prob:
                a_star = self._query_expert(obs0)
                if a_star is not None:
                    a_star_env = np.asarray(a_star, dtype=np.float32)
                    a_star_scaled = self.policy.scale_action(a_star_env[None, :])[0]

                    self._last_expert_action = a_star_env
                    self._last_expert_action_scaled = a_star_scaled
                    self._last_has_expert = True


        return action, buffer_action

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        # attach expert label info so the replay buffer can store it
        if isinstance(infos, (list, tuple)) and len(infos) > 0:
            infos[0]["has_expert"] = bool(self._last_has_expert)
            if self._last_has_expert and (self._last_expert_action_scaled is not None):
                infos[0]["expert_action"] = self._last_expert_action_scaled.copy()

        return super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    # ---------- P2P extra actor update ----------

    def _q_min(self, obs_t: th.Tensor, act_t: th.Tensor) -> th.Tensor:
        q_vals = self.critic(obs_t, act_t)
        if isinstance(q_vals, (tuple, list)) and len(q_vals) >= 2:
            q1, q2 = q_vals[0], q_vals[1]
        else:
            q1, q2 = q_vals[:, 0], q_vals[:, 1]
        return th.min(q1, q2)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Paper-aligned actor update: one actor loss per update:
            L_actor = L_SAC_actor + lambda(t) * E[ gate * has_expert * ||a_pi - a_star||^2 ]
        where gate = sigmoid((Q(s,a_star) - Q(s,a_pi)) / tau).
        BC stays as MSE (as you requested), but is combined into the same backward/step.
        """
        # Set train mode
        self.policy.set_training_mode(True)

        # Update LR schedules (SB3 style)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        if self.ent_coef_optimizer is not None:
            self._update_learning_rate([self.ent_coef_optimizer])

        for gradient_step in range(gradient_steps):
            # --- Sample batch WITH indices so we can pull expert labels ---
            replay_data, batch_inds = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env, return_indices=True
            )

            obs_t = replay_data.observations
            next_obs_t = replay_data.next_observations

            if self.ent_coef_optimizer is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                if hasattr(self, "ent_coef_tensor"):
                    ent_coef = self.ent_coef_tensor
                else:
                    ent_coef = th.tensor(float(self.ent_coef), device=self.device)


            # =========================
            # 1) Critic update (standard SAC)
            # =========================
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(next_obs_t)
                if next_log_prob.ndim == 1:
                    next_log_prob = next_log_prob.unsqueeze(1)

                next_q1, next_q2 = self.critic_target(next_obs_t, next_actions)
                next_q = th.min(next_q1, next_q2) - ent_coef * next_log_prob

                # target: r + gamma * (1-done) * next_q
                target_q = replay_data.rewards + (1.0 - replay_data.dones) * self.gamma * next_q

            current_q1, current_q2 = self.critic(obs_t, replay_data.actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # =========================
            # 2) Actor update (SAC + gated MSE-BC in ONE step)
            # =========================
            for p in self.critic.parameters():
                p.requires_grad = False
            pi_action, log_prob = self.actor.action_log_prob(obs_t)
            if log_prob.ndim == 1:
                log_prob = log_prob.unsqueeze(1)

            q1_pi, q2_pi = self.critic(obs_t, pi_action)
            q_pi = th.min(q1_pi, q2_pi)

            # Standard SAC actor loss
            actor_loss_sac = (ent_coef * log_prob - q_pi).mean()

            # --- Add P2P gated MSE-BC term (paper-style single objective) ---
            lam = self._bc_lambda()
            bc_term = th.tensor(0.0, device=self.device)

            if lam > 0.0 and hasattr(self.replay_buffer, "has_expert"):
                # has_expert: (B,1) for n_envs=1
                has = th.as_tensor(self.replay_buffer.has_expert[batch_inds], device=self.device).float()
                if has.ndim == 1:
                    has = has.unsqueeze(1)

                # expert_actions: (B,1,A) -> (B,A)
                a_star = th.as_tensor(self.replay_buffer.expert_actions[batch_inds], device=self.device).float()
                if a_star.ndim == 3:
                    a_star = a_star.squeeze(1)

                if has.sum().item() > 0:
                    # Gate uses critic comparisons, but we do NOT backprop through the gate (paper-style)
                    with th.no_grad():
                        q1_star, q2_star = self.critic(obs_t, a_star)
                        q_star = th.min(q1_star, q2_star)

                        # ensure (B,1) shapes for gate
                        if q_star.ndim == 1:
                            q_star = q_star.unsqueeze(1)
                        if q_pi.ndim == 1:
                            q_pi_gate = q_pi.unsqueeze(1)
                        else:
                            q_pi_gate = q_pi.detach()

                        tau = max(1e-6, float(self.qfilter_tau))
                        # gate = th.sigmoid((q_star - q_pi_gate) / tau)
                        if self.num_timesteps < self.qfilter_warmup_steps:
                            gate = th.ones_like(q_star)
                        else:
                            gate = th.sigmoid((q_star - q_pi) / tau)


                    # MSE-BC (your choice): compare pi_action vs expert action (both in buffer/scaled action space)
                    bc = ((pi_action - a_star) ** 2).sum(dim=1, keepdim=True)
                    bc_term = (gate * has * bc).mean()

            actor_loss = actor_loss_sac + lam * bc_term

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            for p in self.critic.parameters():
                p.requires_grad = True


            # =========================
            # 3) Entropy temperature update (standard SAC, if enabled)
            # =========================
            if self.ent_coef_optimizer is not None:
                # optimize log(alpha)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # =========================
            # 4) Target network update (standard SAC)
            # =========================
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

                # Handle batch-norm stats if present (SB3 does this internally)
                try:
                    polyak_update(self.critic.batch_norm_stats, self.critic_target.batch_norm_stats, 1.0)
                except Exception:
                    pass

            # Logging (optional but useful)
            self.logger.record("train/critic_loss", float(critic_loss.detach().cpu().item()))
            self.logger.record("train/actor_loss_sac", float(actor_loss_sac.detach().cpu().item()))
            self.logger.record("train/p2p_bc_lambda", float(lam))
            self.logger.record("train/p2p_bc_term", float(bc_term.detach().cpu().item()))
            self.logger.record("train/actor_loss_total", float(actor_loss.detach().cpu().item()))

