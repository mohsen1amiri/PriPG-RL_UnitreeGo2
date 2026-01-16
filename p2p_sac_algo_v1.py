import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer


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
        qfilter_tau: float = 1.0,         # sigmoid temperature
        bc_gradient_steps: int = 1,        # extra BC updates per SAC train() step
        **kwargs,
    ):
        self.prefill_steps = int(prefill_steps)

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
            if np.random.rand() < self._p_expert():
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
            if np.random.rand() < self.label_prob:
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
        # normal SAC updates
        super().train(gradient_steps, batch_size)

        lam = self._bc_lambda()
        if lam <= 0.0:
            return

        # extra Q-filtered BC updates
        if not hasattr(self.replay_buffer, "has_expert"):
            return

        for _ in range(max(1, self.bc_gradient_steps)):
            replay_data, batch_inds = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env, return_indices=True
            )

            has = th.as_tensor(self.replay_buffer.has_expert[batch_inds], device=self.device)  # (B,1)
            if has.ndim == 2:
                pass
            elif has.ndim == 3:
                has = has.squeeze(1)

            a_star = th.as_tensor(self.replay_buffer.expert_actions[batch_inds], device=self.device)  # (B,1,A)
            if a_star.ndim == 3:
                a_star = a_star.squeeze(1)  # -> (B,A)


            obs_t = replay_data.observations

            # sample policy action (reparameterized) for actor loss
            pi_action, _log_prob = self.actor.action_log_prob(obs_t)

            with th.no_grad():
                q_pi = self._q_min(obs_t, pi_action)     # (B,1) or (B,)
                q_star = self._q_min(obs_t, a_star)

                # make shapes consistent
                if q_pi.ndim == 1:
                    q_pi = q_pi.unsqueeze(1)
                if q_star.ndim == 1:
                    q_star = q_star.unsqueeze(1)

                gate = th.sigmoid((q_star - q_pi) / max(1e-6, self.qfilter_tau))  # (B,1)

            # BC loss: MSE between sampled pi_action and expert action
            bc = ((pi_action - a_star) ** 2).sum(dim=1, keepdim=True)  # (B,1)

            loss = lam * (gate * has * bc).mean()

            self.actor.optimizer.zero_grad()
            loss.backward()
            self.actor.optimizer.step()

            # log
            self.logger.record("train/p2p_bc_lambda", float(lam))
            self.logger.record("train/p2p_bc_loss", float(loss.detach().cpu().item()))
