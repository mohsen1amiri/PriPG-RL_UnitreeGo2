import warnings
from typing import Any, Optional, TypeVar, Union, Callable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torch as th
from torch.nn import functional as F
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from on_policy_algorithm_custom import OnPolicyAlgorithm
from Buffer_Custom import RolloutBuffer, DictRolloutBuffer

SelfPPO = TypeVar("SelfPPO", bound="PPO_MPC")

# policy_kwargs = dict(net_arch=[50, 50])
# policy_kwargs = dict(net_arch=[256, 256, 256])

class PPO_MPC(OnPolicyAlgorithm):
    """
    PPO with an additional imitation/MPC loss term.

    Extra arg:
        imitation_target_fn(obs_batch, actions_batch, infos_batch)
            -> tensor/np.ndarray of expert actions.
        If None, we assume infos_batch[i]["Action MPC"] exists (original behaviour).
    """

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.98,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # Additional MPC-related parameters:
        Adaptive_Beta: bool = True,
        Just_Beta: bool = True,
        MSE_Weight: float = 1.0,
        Initial_Beta: float = 1.0,
        start_iteration_number: int = 0,
        end_iteration_number: int = 10,
        # Auxiliary imitation function:
        imitation_target_fn: Optional[
            Callable[[th.Tensor, th.Tensor, Sequence[dict[str, Any]]], Union[th.Tensor, np.ndarray]]
        ] = None,
    ) -> None:

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,   # we call _setup_model below if requested
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self._n_updates = 0

        # MPC/imitation parameters
        self.adaptive_beta = Adaptive_Beta
        self.hu_weight = MSE_Weight
        self.initial_beta = Initial_Beta
        self.just_beta = Just_Beta
        self.sp = start_iteration_number / n_steps - 1
        self.ep = end_iteration_number / n_steps - 1
        self.imitation_target_fn = imitation_target_fn

        # ---- Make imitation schedule & beta readable outside train() ----
        self.beta = float(Initial_Beta)          # exists even before first train()
        self.current_beta = self.beta            # stable name for callbacks/inspection

        # Store original timestep thresholds (you said end_iteration_number is TIMESTEPS)
        self.start_timesteps = int(start_iteration_number)
        self.end_timesteps = int(end_iteration_number)

        # Track if we already disabled expert labels in the env
        self._expert_disabled = False


        # history of positions for plotting in learn()
        self.position_history: list[np.ndarray] = []




        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1, (
                "`n_steps * n_envs` must be > 1. "
                f"Currently n_steps={self.n_steps}, n_envs={self.env.num_envs}"
            )
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size}, "
                    f"but the RolloutBuffer is of size {buffer_size} (= n_steps * n_envs). "
                    f"After every {untruncated_batches} full batches, there will be "
                    f"a truncated batch of size {buffer_size % batch_size}."
                )

        if _init_setup_model:
            self._setup_model()

    # ------------------------------------------------------------------ #
    #  Setup model
    # ------------------------------------------------------------------ #
    def _setup_model(self) -> None:
        super()._setup_model()
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive or None."
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _disable_expert_labels_in_env(self) -> None:
        """
        Turns off expert-label generation inside the env (collect_expert/planner).
        Works with VecEnv/Monitor wrapping.
        """
        if self.env is None:
            return

        env = self.env
        env_list = env.envs if hasattr(env, "envs") else [env]

        for e in env_list:
            base = e
            for _ in range(10):
                if hasattr(base, "env"):
                    base = base.env
                    continue
                if hasattr(base, "unwrapped"):
                    base = base.unwrapped
                break

            if hasattr(base, "collect_expert"):
                base.collect_expert = False
            if hasattr(base, "planner"):
                base.planner = None


    # ------------------------------------------------------------------ #
    #  Train step (like PPO, plus imitation term)
    # ------------------------------------------------------------------ #
    def train(self) -> None:
        # Switch to train mode
        self.policy.set_training_mode(True)
        # Update LR
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        td_losses, imitation_losses, main_losses = [], [], []
        entropy_losses, pg_losses, value_losses = [], [], []
        clip_fractions, all_kl_divs = [], []

        # How many rollouts we completed so far
        completed_rollouts = self._n_updates // self.n_epochs

        # β schedule
        if self.adaptive_beta:
            if completed_rollouts <= self.sp:
                self.beta = self.initial_beta
            elif completed_rollouts >= self.ep:
                self.beta = 0.0
            else:
                frac = (completed_rollouts - self.sp) / float(self.ep - self.sp)
                self.beta = self.initial_beta * (1.0 - frac)
        else:
            self.beta = self.initial_beta

        continue_training = True

        # ---- Decide whether imitation is active ----
        # Condition A: beta is effectively zero
        beta_off = (self.beta <= 0.0)

        # Condition B: timesteps have reached end_timesteps (your meaning of end_iteration_number)
        # self.num_timesteps exists in SB3 OnPolicyAlgorithm and counts env steps
        time_off = (self.num_timesteps >= self.end_timesteps)

        # Special case: your "pure imitation" hack
        if self.initial_beta == 100000:
            use_imitation = True
        else:
            use_imitation = (not beta_off) and (not time_off)

        # Publish beta so external code/callbacks can read it
        self.current_beta = float(self.beta)

        # --- Effective beta: if imitation is OFF, force beta=0 for mixing & KL logic ---
        effective_beta = float(self.beta) if use_imitation else 0.0

        # Keep a stable “what we actually used” value for logging/callbacks
        self.current_beta = effective_beta

        # If imitation is off, disable expert labels in env ONCE (saves huge compute)
        if (not use_imitation) and (not self._expert_disabled):
            self._disable_expert_labels_in_env()
            self._expert_disabled = True
            if self.verbose >= 1:
                print(f">> [PPO_MPC] Expert labels disabled at t={self.num_timesteps}, beta={self.beta}")


        # Epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Full pass over rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                obs_batch = rollout_data.observations
                values_old = rollout_data.old_values
                logp_old = rollout_data.old_log_prob
                advantages = rollout_data.advantages
                returns = rollout_data.returns
                infos_batch = rollout_data.infos
                actions = rollout_data.actions

                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                #if self.use_sde:
                    # self.policy.reset_noise(self.batch_size)
                if self.use_sde:
                    self.policy.reset_noise(len(obs_batch))


                values, log_prob, entropy = self.policy.evaluate_actions(obs_batch, actions)
                values = values.flatten()

                # Advantage normalization
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # PPO ratio
                ratio = th.exp(log_prob - logp_old)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss (with optional clipping)
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = values_old + th.clamp(values - values_old, -clip_range_vf, clip_range_vf)

                value_loss = F.mse_loss(returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Base TD loss
                # ----- Split TD into actor vs critic parts -----
                td_actor_loss = policy_loss + self.ent_coef * entropy_loss
                td_critic_loss = self.vf_coef * value_loss
                td_loss = td_actor_loss + td_critic_loss
                td_losses.append(td_loss.item())


                # Approx KL for early stopping
                with th.no_grad():
                    log_ratio = log_prob - logp_old
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl and (not use_imitation):
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to KL={approx_kl_div:.2f}")
                    break

                # ------------------ imitation / MPC target -------------------
                # ------------------ imitation / MPC target (ONLY when active) -------------------
                if use_imitation:
                    if self.imitation_target_fn is not None:
                        expert_actions_raw = self.imitation_target_fn(obs_batch, actions, infos_batch)
                        expert_actions = th.as_tensor(expert_actions_raw, device=self.device)
                    else:
                        # original behaviour: read "Action MPC" from infos
                        expert_actions = th.tensor(
                            [info["Action MPC"] for info in infos_batch],
                            device=self.device,
                        )

                    if isinstance(self.action_space, spaces.Discrete):
                        expert_actions = expert_actions.long().flatten()

                    dist = self.policy.get_distribution(obs_batch)
                    expert_actions = expert_actions.to(self.device).float()
                    log_prob_expert = dist.log_prob(expert_actions)
                    imitation_loss = -log_prob_expert.mean()
                else:
                    imitation_loss = th.zeros((), device=self.device)

                imitation_losses.append(float(imitation_loss.detach().cpu().item()))





                # Combine losses using your original Beta logic
                # ----- Stable mixing: always train critic, always train PPO actor, add imitation as actor-regularizer -----


                if self.initial_beta == 100000:
                    loss = self.hu_weight * imitation_loss
                else:
                    if self.just_beta:
                        if effective_beta == 1.0:
                            loss = (self.hu_weight * imitation_loss) + td_critic_loss + td_actor_loss
                        elif effective_beta == 0.0:
                            loss = td_critic_loss + td_actor_loss
                        else:
                            loss = (self.hu_weight * effective_beta * imitation_loss) + td_actor_loss + td_critic_loss
                    else:
                        if effective_beta == 1.0:
                            loss = self.hu_weight * imitation_loss + td_critic_loss
                        elif effective_beta == 0.0:
                            loss = td_actor_loss + td_critic_loss
                        else:
                            loss = (self.hu_weight * effective_beta * imitation_loss) + (1 - effective_beta) * td_actor_loss + td_critic_loss


                main_losses.append(loss.item())

                # Optimizer step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs

        # ------------------ position logging -------------------
        all_obs = self.rollout_buffer.observations
        position = None

        if isinstance(all_obs, dict):
            # This branch is for dict observations (other envs).
            # Example if you had root position as "root_pos_w":
            pos_key = "root_pos_w"  # only used for dict-type obs
            if pos_key in all_obs:
                position = all_obs[pos_key]
        else:
            # Your Go2MPCEnv case: obs is a flat array [x, y]
            # rollouts: shape (n_steps * n_envs, 2)
            position = all_obs  # already [x, y]

        if position is not None:
            # Ensure we only keep x,y for logging/plots
            if position.shape[1] >= 2:
                pos_xy = position[:, :2]
            else:
                pos_xy = position

            # store for plotting in learn()
            self.position_history.append(pos_xy.copy())



        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # --- log everything (unchanged) ---
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/td_loss", np.mean(td_losses))
        self.logger.record("train/imitation_loss", np.mean(imitation_losses))
        self.logger.record("train/Main_loss", np.mean(main_losses))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record("train/Beta", self.current_beta)




    # ------------------------------------------------------------------ #
    #  Learn override: after training, dump I_meas CSV + plot
    # ------------------------------------------------------------------ #
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO_MPC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:

        model = super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

        # After training: save robot position history if any
        if len(self.position_history) > 0:
            full_pos = np.concatenate(self.position_history, axis=0)  # (N, 2)

            df = pd.DataFrame(full_pos, columns=["x", "y"])
            from pathlib import Path

            log_dir = Path(self.logger.get_dir())
            run_name = log_dir.name
            weights_dir = log_dir / "Weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            csv_path = weights_dir / f"robot_position_full_buffer_{run_name}.csv"
            df.to_csv(csv_path, index=False)

            x = full_pos[:, 0]
            y = full_pos[:, 1]

            N = x.shape[0]
            time_idx = np.arange(N)
            colors = time_idx / float(N - 1)

            fig, ax = plt.subplots()
            sc = ax.scatter(
                x,
                y,
                c=colors,
                cmap="viridis",
                s=5,
                alpha=0.8,
            )
            ax.set_xlabel("x position")
            ax.set_ylabel("y position")
            ax.set_title("Robot base position over training")
            fig.colorbar(sc, ax=ax, label="progress through training")

            # circle = Circle((0.0, 0.0), radius=30.0, edgecolor="red", fill=False, linewidth=2)
            # ax.add_patch(circle)

            plots_dir = log_dir / "Weights"
            plots_dir.mkdir(parents=True, exist_ok=True)
            fig_path = plots_dir / f"robot_position_train_{run_name}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        return model
