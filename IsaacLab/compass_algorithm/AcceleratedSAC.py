import torch
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

class AcceleratedSAC(SAC):
    """
    Implementation of 'Accelerating actor-critic-based algorithms via pseudo-labels' 
    (Beikmohammadi & Magnússon, 2024), modified with a Plateau-then-Decay schedule.
    """
    def __init__(
        self,
        *args,
        beta_0: float = 1.0,         # Initial weight for the imitation loss
        plateau_steps: int = 100_000, # Steps to hold beta_e = beta_0
        decay_steps: int = 50_000,   # Steps to decay beta_e to 0 AFTER the plateau
        c2_noise: float = 0.01,       # Noise added to expert action (c_2 in Eq 7)
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        # Accelerated SAC Hyperparameters
        self.beta_0 = beta_0
        self.plateau_steps = plateau_steps
        self.decay_steps = decay_steps
        self.c2_noise = c2_noise

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        if self.ent_coef_optimizer is not None:
            self._update_learning_rate([self.ent_coef_optimizer])

        for gradient_step in range(gradient_steps):
            # 1. Sample from our custom buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            expert_actions = getattr(replay_data, 'expert_actions', replay_data.actions)
            has_expert = getattr(replay_data, 'has_expert', torch.zeros_like(replay_data.rewards))

            obs = replay_data.observations
            next_obs = replay_data.next_observations

            ent_coef = torch.exp(self.log_ent_coef.detach()) if self.ent_coef_optimizer is not None else self.ent_coef_tensor

            # ==========================================================
            # 1. STANDARD CRITIC UPDATE (NO CQL)
            # ==========================================================
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                next_q1, next_q2 = self.critic_target(next_obs, next_actions)
                next_q = torch.min(next_q1, next_q2) - ent_coef * next_log_prob.unsqueeze(-1)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q

            current_q1, current_q2 = self.critic(obs, replay_data.actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # ==========================================================
            # 2. ACCELERATED ACTOR UPDATE
            # ==========================================================
            # Standard SAC Actor Loss
            pi_actions, log_prob = self.actor.action_log_prob(obs)
            q1_pi, q2_pi = self.critic(obs, pi_actions)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss_sac = (ent_coef * log_prob.unsqueeze(-1) - q_pi).mean()
            
            # --- PLATEAU-THEN-DECAY MATH START ---
            if self.num_timesteps <= self.plateau_steps:
                # Hold at maximum weight
                beta_e = self.beta_0
            else:
                # Calculate how far we are into the decay phase (0.0 to 1.0)
                decay_progress = min(1.0, (self.num_timesteps - self.plateau_steps) / self.decay_steps)
                beta_e = max(0.0, (1.0 - decay_progress) * self.beta_0)
            
            # Apply pseudo-label noise
            if self.c2_noise > 0.0:
                noise = torch.clamp(torch.randn_like(expert_actions) * self.c2_noise, -1.0, 1.0)
                expert_actions_noisy = torch.clamp(expert_actions + noise, -2.0, 2.0)
            else:
                expert_actions_noisy = expert_actions

            # --- NECESSARY ENGINEERING FIX (Normalizes scale without changing theory) ---
            action_low = torch.tensor(self.action_space.low, device=self.device)
            action_high = torch.tensor(self.action_space.high, device=self.device)
            
            # Scale expert from [-2, 2] down to [-1, 1] to match pi_actions
            normalized_expert_noisy = 2.0 * (expert_actions_noisy - action_low) / (action_high - action_low) - 1.0
            # ----------------------------------------------------------------------------

            # Calculate Quadratic Penalty (0.5 * beta_e * MSE)
            mse_loss = F.mse_loss(pi_actions, normalized_expert_noisy, reduction='none').mean(dim=1, keepdim=True)
            imitation_loss = (mse_loss * has_expert).mean()
            
            # Combine losses
            # actor_loss = (1-beta_e) * actor_loss_sac +  beta_e * imitation_loss
            actor_loss = actor_loss_sac +  beta_e * imitation_loss
            # --- PLATEAU-THEN-DECAY MATH END ---

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # ==========================================================
            # 3. ENTROPY AND TARGET UPDATES
            # ==========================================================
            if self.ent_coef_optimizer is not None:
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                try:
                    polyak_update(self.critic.batch_norm_stats, self.critic_target.batch_norm_stats, 1.0)
                except Exception:
                    pass

            # ==========================================================
            # 4. LOGGING
            # ==========================================================
            self.logger.record("train/actor_loss_sac", float(actor_loss_sac.item()))
            self.logger.record("train/imitation_loss", float(imitation_loss.item()))
            self.logger.record("train/beta_e", float(beta_e))