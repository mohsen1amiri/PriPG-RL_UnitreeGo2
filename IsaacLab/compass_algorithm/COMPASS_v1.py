import torch
import torch.nn.functional as F
import numpy as np
import math
from collections import deque
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.type_aliases import ReplayBufferSamples

class COMPASS_v1(SAC):
    """
    Conservative Planner-guided Soft Actor-Critic (CP2P-SAC)
    """
    def __init__(
        self,
        *args,       
        anchor_weight: float = 1.0,        
        gate_tau: float = 1.0,             
        # --- NEW: Bernstein Maturation Hyperparameters ---
        stationarity_tolerance: float = 5e-3,  # Epsilon: The True Risk target
        patience_window: int = 100_000,       # K: Steps required to prove mastery
        confidence_delta: float = 0.01,    # Delta: 95% confidence interval
        loss_range_b: float = 4.0,         # b: Max theoretical MSE range [-2,2]^2
        burn_in_steps: int = 100_000,
        ablation_maturation: str = "none",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        # CP2P-SAC Hyperparameters
        self.anchor_weight = anchor_weight
        self.gate_tau = gate_tau
        
        # Bernstein Hyperparameters
        self.stationarity_tolerance = stationarity_tolerance
        self.patience_window = patience_window
        self.confidence_delta = confidence_delta
        self.loss_range_b = loss_range_b
        
        # Bernstein Constant 'c' = sqrt(2) for standard concentration inequality
        self.bernstein_c = math.sqrt(2.0) 
        
        # Initialize Sliding Window and Absorbing State
        self.nll_window = deque(maxlen=self.patience_window)
        self.is_mature = 0.0  # M_t
        self.ucb_window = deque(maxlen=self.patience_window)
        self.trend_slope = 0.0
        
        # Initialize variables for comprehensive logging
        self.current_ucb = 0.0 
        self.mu_k = 0.0
        self.sigma_sq_k = 0.0
        self.var_penalty = 0.0
        self.scale_penalty = 0.0
        self.mean_nll_batch = 0.0

        self.burn_in_steps = burn_in_steps
        self.ablation_maturation = ablation_maturation


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
            # 1. STANDARD CRITIC UPDATE (CQL REMOVED)
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
            # 2. ABSORBING MATURITY ASSESSMENT (Empirical Bernstein Bound)
            # ==========================================================
            for param in self.critic.parameters():
                param.requires_grad = False

            pi_actions, log_prob = self.actor.action_log_prob(obs)
            
            # pi_mean = self.actor.get_action_dist_params(obs)[0] 
            # squashed_pi_mean = torch.tanh(pi_mean)
            
            # # --- ACTION SCALING FIX ---
            # # Inverse scale the expert actions from [low, high] down to [-1.0, 1.0]
            # action_low = torch.tensor(self.action_space.low, device=self.device)
            # action_high = torch.tensor(self.action_space.high, device=self.device)
            
            # normalized_expert = 2.0 * (expert_actions - action_low) / (action_high - action_low) - 1.0

            # normalized_expert = torch.clamp(normalized_expert, -0.999, 0.999)
            
            # # Calculate the loss against the properly scaled expert
            # nll_loss = F.mse_loss(squashed_pi_mean, normalized_expert, reduction='none').mean(dim=1, keepdim=True)


            # Get the RAW unbounded Gaussian mean from the network
            pi_mean = self.actor.get_action_dist_params(obs)[0] 
            squashed_pi_mean = torch.tanh(pi_mean) # We need this for the evaluation
            
            # --- ACTION SCALING ---
            action_low = torch.tensor(self.action_space.low, device=self.device)
            action_high = torch.tensor(self.action_space.high, device=self.device)
            
            normalized_expert = 2.0 * (expert_actions - action_low) / (action_high - action_low) - 1.0
            clamped_expert = torch.clamp(normalized_expert, -0.9999, 0.9999) 
            
            # ==========================================================
            # A. OPTIMIZATION LOSS (Logit-Space for fast, strong gradients)
            # ==========================================================
            expert_logits = torch.atanh(clamped_expert)
            optim_loss = F.mse_loss(pi_mean, expert_logits, reduction='none').mean(dim=1, keepdim=True)
            
            # ==========================================================
            # B. EVALUATION LOSS (Bounded Physical-Space for meaningful UCB)
            # ==========================================================
            eval_loss = F.mse_loss(squashed_pi_mean, clamped_expert, reduction='none').mean(dim=1, keepdim=True)
            
            # --- UPDATE UCB WINDOW WITH THE EVALUATION LOSS ---
            valid_experts = has_expert > 0
            if valid_experts.any():
                # Extract clean scalar means for the valid expert transitions
                self.mean_eval_batch = eval_loss[valid_experts].detach().mean().item()
                self.mean_optim_batch = optim_loss[valid_experts].detach().mean().item()
                
                # Append ONLY the physical evaluation loss to the Bernstein Window
                self.nll_window.append(self.mean_eval_batch)
                
                if len(self.nll_window) == self.patience_window:
                    self.mu_k = float(np.mean(self.nll_window))
                    self.sigma_sq_k = float(np.var(self.nll_window))
                    
                    log_term = math.log(1.0 / self.confidence_delta)
                    self.var_penalty = self.bernstein_c * math.sqrt((self.sigma_sq_k * log_term) / self.patience_window)
                    self.scale_penalty = (self.loss_range_b * log_term) / (3.0 * self.patience_window)
                    
                    self.current_ucb = self.mu_k + self.var_penalty + self.scale_penalty
                    self.ucb_window.append(self.current_ucb)
                    
                    # --- NEW: OLS Trend Slope Math (Eq. 5 in paper) ---
                    if len(self.ucb_window) == self.patience_window:
                        K = self.patience_window
                        tau = np.arange(1, K + 1)
                        tau_mean = (K + 1) / 2.0
                        multiplier = 12.0 / (K * (K**2 - 1))
                        
                        ucb_array = np.array(self.ucb_window)
                        self.trend_slope = float(multiplier * np.sum((tau - tau_mean) * ucb_array))
                        
                        # --- MATURATION LOGIC ---
                        if self.ablation_maturation == "naive_decay":
                            indicator = 1.0 if self.num_timesteps >= self.burn_in_steps else 0.0
                        else:
                            if self.num_timesteps >= max(self.patience_window, self.burn_in_steps):
                                # Mature if slope is flat (between slightly negative and strictly 0)
                                if -self.stationarity_tolerance <= self.trend_slope <= 0.0:
                                    indicator = 1.0
                                else:
                                    indicator = 0.0
                            else:
                                indicator = 0.0  # Locked during burn-in
                        
                        if indicator == 1.0 and self.is_mature == 0.0:
                            reason = "NAIVE DECAY" if self.ablation_maturation == "naive_decay" else f"SLOPE: {self.trend_slope:.7f}"
                            print(f"\n[COMPASS] Agent MATURED at step {self.num_timesteps}! ({reason})")
                            
                        self.is_mature = max(self.is_mature, indicator)
                
            # This broadcasts the current timestep and maturity to the physical environment
            if hasattr(self.env, "env_method"):
                self.env.env_method("set_compass_status", float(self.is_mature), int(self.num_timesteps))

            # ==========================================================
            # 3. GATED ACTOR UPDATE
            # ==========================================================
            q1_pi, q2_pi = self.critic(obs, pi_actions)
            q_pi = torch.min(q1_pi, q2_pi)
            
            actor_loss_sac = (ent_coef * log_prob.unsqueeze(-1) - q_pi).mean()
            
            with torch.no_grad():
                q1_star, q2_star = self.critic(obs, expert_actions)
                q_star = torch.min(q1_star, q2_star)
                
                delta_phi = q_star - q_pi
                m_phi = torch.sigmoid(delta_phi / self.gate_tau)
            
            G_phi = (1.0 - self.is_mature) * 1.0 + self.is_mature * m_phi
            anchor_loss = (G_phi * optim_loss * has_expert).mean()
            
            # PURE BC WARM-UP LOGIC
            actor_loss = (self.is_mature * actor_loss_sac) + self.anchor_weight * anchor_loss

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True

            # ==========================================================
            # 4. ENTROPY AND TARGET UPDATES
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
            # 5. COMPREHENSIVE TENSORBOARD LOGGING
            # ==========================================================
            # A. Neural Network / Gradients
            self.logger.record("train/gate_mean_G_phi", float(G_phi.mean().item()))
            
            # B. The Twin Behavior Cloning Losses
            if valid_experts.any():
                self.logger.record("train/loss_optim_logit", float(self.mean_optim_batch))
                self.logger.record("train/loss_eval_physical", float(self.mean_eval_batch))
                self.logger.record("train/loss_anchor_total", float(anchor_loss.item()))
            
            # C. The Bernstein Maturation Math
            self.logger.record("maturity/1_window_mean_mu", float(self.mu_k))
            self.logger.record("maturity/2_window_variance_sigma2", float(self.sigma_sq_k))
            self.logger.record("maturity/3_variance_penalty", float(self.var_penalty))
            self.logger.record("maturity/4_bernstein_ucb", float(self.current_ucb))
            self.logger.record("maturity/5_is_mature_M", float(self.is_mature))
            self.logger.record("maturity/5b_trend_slope_m", float(self.trend_slope))