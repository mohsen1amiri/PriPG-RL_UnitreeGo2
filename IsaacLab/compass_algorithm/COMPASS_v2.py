import torch
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

class COMPASS_v2(SAC):
    """
    Conservative Planner-guided Soft Actor-Critic (CP2P-SAC)
    Uses a deterministic Plateau-Decay schedule for the expert weight, 
    and unlocks the Advantage Q-Filter Gate only after the decay is complete.
    """
    def __init__(
        self,
        *args,       
        beta_0: float = 10.0,             # Initial heavy weight during plateau
        beta_final: float = 1.0,          # Final residual weight after decay
        plateau_steps: int = 200_000,     # Steps to hold beta = beta_0
        decay_steps: int = 500_000,       # Steps to decay beta down to beta_final
        gate_tau: float = 1.0,            # Temperature for the Q-Filter sigmoid
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        # Schedule Hyperparameters
        self.beta_0 = beta_0
        self.beta_final = beta_final
        self.plateau_steps = plateau_steps
        self.decay_steps = decay_steps
        
        self.gate_tau = gate_tau
        
        # Absorbing state for the Gate
        self.is_mature = 0.0  # M_t
        
        # For logging
        self.mean_optim_batch = 0.0
        self.current_beta = beta_0

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        if self.ent_coef_optimizer is not None:
            self._update_learning_rate([self.ent_coef_optimizer])

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            expert_actions = getattr(replay_data, 'expert_actions', replay_data.actions)
            has_expert = getattr(replay_data, 'has_expert', torch.zeros_like(replay_data.rewards))

            obs = replay_data.observations
            next_obs = replay_data.next_observations

            ent_coef = torch.exp(self.log_ent_coef.detach()) if self.ent_coef_optimizer is not None else self.ent_coef_tensor

            # ==========================================================
            # 1. STANDARD CRITIC UPDATE
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
            # 2. DETERMINISTIC 3-PHASE SCHEDULE
            # ==========================================================
            if self.num_timesteps <= self.plateau_steps:
                # Phase 1: Heavy Imitation
                self.current_beta = self.beta_0
                self.is_mature = 0.0
                
            elif self.num_timesteps <= self.plateau_steps + self.decay_steps:
                # Phase 2: Smooth Decay
                decay_progress = (self.num_timesteps - self.plateau_steps) / self.decay_steps
                self.current_beta = self.beta_0 - decay_progress * (self.beta_0 - self.beta_final)
                self.is_mature = 0.0
                
            else:
                # Phase 3: Mature RL Fine-Tuning
                self.current_beta = self.beta_final
                
                if self.is_mature == 0.0:
                    print(f"\n[COMPASS] 🚀 Agent MATURED at step {self.num_timesteps}! Unlocking Q-Filter Gate.")
                    self.is_mature = 1.0

            # Send maturity to the environment (to bypass REAP planner logic)
            if hasattr(self.env, "env_method"):
                self.env.env_method("set_compass_status", float(self.is_mature), int(self.num_timesteps))


            # ==========================================================
            # 3. LOGIT-SPACE ANCHOR LOSS (No Vanishing Gradients)
            # ==========================================================
            for param in self.critic.parameters():
                param.requires_grad = False

            pi_actions, log_prob = self.actor.action_log_prob(obs)
            
            # Get the RAW unbounded Gaussian mean from the network
            pi_mean = self.actor.get_action_dist_params(obs)[0] 
            
            # Inverse scale the expert actions from [low, high] down to [-1.0, 1.0]
            action_low = torch.tensor(self.action_space.low, device=self.device)
            action_high = torch.tensor(self.action_space.high, device=self.device)
            
            normalized_expert = 2.0 * (expert_actions - action_low) / (action_high - action_low) - 1.0
            clamped_expert = torch.clamp(normalized_expert, -0.9999, 0.9999) 
            
            # Map physical expert into Logit Space
            expert_logits = torch.atanh(clamped_expert)
            optim_loss = F.mse_loss(pi_mean, expert_logits, reduction='none').mean(dim=1, keepdim=True)
            
            valid_experts = has_expert > 0
            if valid_experts.any():
                self.mean_optim_batch = optim_loss[valid_experts].detach().mean().item()

            # ==========================================================
            # 4. GATED ACTOR UPDATE
            # ==========================================================
            q1_pi, q2_pi = self.critic(obs, pi_actions)
            q_pi = torch.min(q1_pi, q2_pi)
            
            actor_loss_sac = (ent_coef * log_prob.unsqueeze(-1) - q_pi).mean()
            
            with torch.no_grad():
                q1_star, q2_star = self.critic(obs, expert_actions)
                q_star = torch.min(q1_star, q2_star)
                
                delta_phi = q_star - q_pi
                m_phi = torch.sigmoid(delta_phi / self.gate_tau)
            
            # The Gate: 
            # When M_t=0 (Phases 1 & 2), G_phi is strictly 1.0 (Pure BC).
            # When M_t=1 (Phase 3), G_phi equals m_phi (Dynamic Q-Filter).
            G_phi = (1.0 - self.is_mature) * 1.0 + self.is_mature * m_phi
            
            # Anchor loss combines the current Beta weight and the Gate status
            anchor_loss = (self.current_beta * G_phi * optim_loss * has_expert).mean()
            
            # Final additive loss (Entropy is perfectly protected inside actor_loss_sac)
            actor_loss = actor_loss_sac + anchor_loss

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True

            # ==========================================================
            # 5. ENTROPY AND TARGET UPDATES
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
            # 6. COMPREHENSIVE TENSORBOARD LOGGING
            # ==========================================================
            self.logger.record("train/schedule_beta_e", float(self.current_beta))
            self.logger.record("train/schedule_M_t", float(self.is_mature))
            self.logger.record("train/gate_mean_G_phi", float(G_phi.mean().item()))
            
            if valid_experts.any():
                self.logger.record("train/loss_optim_logit", float(self.mean_optim_batch))
                self.logger.record("train/loss_anchor_total", float(anchor_loss.item()))