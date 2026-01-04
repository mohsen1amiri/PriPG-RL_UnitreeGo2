import os, pathlib
home = pathlib.Path.home()

# Point Kit/Omniverse caches to $HOME (works well on clusters)
os.environ.setdefault("XDG_CACHE_HOME", str(home / ".cache"))
os.environ.setdefault("OMNI_CACHE_ROOT", str(home / ".cache" / "ov"))
os.environ.setdefault("OMNI_KIT_CACHE_DIR", str(home / ".cache" / "ov" / "Kit"))
os.environ.setdefault("USD_CACHEDIR", str(home / ".cache" / "usd"))

# Optional but quiets extra warnings:
(os.environ.setdefault)("PXR_PLUGINPATH_NAME", "")  # if you don’t rely on custom USD plugins
for p in [
    home / ".cache" / "ov" / "Kit",
    home / ".cache" / "usd",
    home / "Documents" / "Kit" / "shared",
]:
    p.mkdir(parents=True, exist_ok=True)

# Headless & safe plotting on HPC
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMNI_KIT_FORCE_HEADLESS", "1")

from isaacsim.simulation_app import SimulationApp
_SIM = SimulationApp({"headless": True})
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))                # for on_policy_algorithm_custom, Buffer_Custom, etc.





import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union
import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3.common import logger
from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer
# from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.logger import configure
from stable_baselines3.ppo.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.utils import get_schedule_fn, get_linear_fn
# import Custom_env.gymnasium_env
import sys
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # use GPU 0; set to '-1' for CPU
import argparse
import pandas as pd
import random
import pylab
import copy
import math
from tensorboardX import SummaryWriter
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Dense
# from tensorflow.keras import backend as K
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(gpus) > 0:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError:
#         pass
from on_policy_algorithm_custom import OnPolicyAlgorithm
from Buffer_Custom import DictRolloutBuffer
import argparse

arg_pass = argparse.ArgumentParser()

arg_pass.add_argument(
    "--seed",
    help='Seed; default 1357924680',
    type=int,  
    default=1357924680,
)
arg_pass.add_argument(
    "--w",
    help='omega, weighting value for the imitation objective; default 1',
    type=float, 
    default=1,
)
arg_pass.add_argument(
    "--e",
    help='end_iteration_number; default 320_400',
    type=int,  
    default=320_400,
)
arg_pass.add_argument(
    "--total_timesteps",
    help='total_timesteps; default 4_806_000',
    type=int,  
    default=4_806_000,
)
arg_pass.add_argument(
    "--n_epochs",
    help='n_epochs; default 10',
    type=int,  
    default=10,
)
arg_pass.add_argument(
    "--start_iteration_number",
    help='start_iteration_number; default 160_200',
    type=int,
    default=160_200,
)
arg_pass.add_argument(
    "--gamma",
    help='gamma; default 0.98',
    type=float,
    default=0.98,
)
arg_pass.add_argument(
    "--Initial_Beta",
    help='Initial_Beta; default 1.0',
    type=float,
    default=1.0,
)
arg_pass.add_argument(
    "--n_steps",
    help='n_steps; default 801',
    type=int,
    default=801,
)
arg_pass.add_argument(
    "--MPC_error_ratio",
    help='MPC_error_ratio; default 0.01',
    type=float,
    default=0.01,
)
arg_pass.add_argument(
    "--MPC_senario",
    help='MPC_senario; default biased_MPC (or "noisey_MPC")',
    type=str,
    default="biased_MPC",
)
arg_pass.add_argument(
    "--safety_const",
    help='safety_const; default True',
    type=str,
    default="True",
)
arg_pass.add_argument(
    "--naive_reward",
    help='naive_reward; default False',
    type=str,
    default="False",
)
arg_pass.add_argument(
    "--margin",
    help='margin; default 0.0',
    type=float,
    default=0.0,
)
arg_pass.add_argument(
    "--learning_rate",
    help="learning_rate; default 1e-4",
    type=float,
    default=1e-4,
)
arg_pass.add_argument(
    "--target_kl",
    help="target_kl; default 0.2",
    type=float,
    default=0.2,
)
arg_pass.add_argument(
    "--ent_coef",
    help="ent_coef; default 0.0",
    type=float,
    default=0.0,
)
arg_pass.add_argument(
    "--clip_range_type",
    help="clip_range_type; default Inc (or Dec or fixed)",
    type=str,
    default="Inc",
)
arg_pass.add_argument(
    "--policy_net_arch",
    help="policy_net_arch; default 20",
    type=int,
    default=20,
)


args = arg_pass.parse_args()

Seed = args.seed
MSE_Weight = args.w
end_iteration_number = args.e
start_iteration_number = args.start_iteration_number
gamma = args.gamma
Initial_Beta = args.Initial_Beta
n_steps = args.n_steps
train_freq = n_steps
total_timesteps = args.total_timesteps
n_epochs = args.n_epochs
MPC_error_ratio = args.MPC_error_ratio
MPC_senario = args.MPC_senario
safety_const = args.safety_const
naive_reward = args.naive_reward
margin = args.margin
learning_rate = args.learning_rate
target_kl = args.target_kl
ent_coef = args.ent_coef
clip_range_type = args.clip_range_type
policy_net_arch = args.policy_net_arch
policy_kwargs = dict(net_arch=[policy_net_arch, policy_net_arch])
print(f"policy_net_arch: {policy_net_arch}, policy_kwargs: {policy_kwargs}")


if safety_const == "True":
    safety_const = True
elif safety_const == "False":
    safety_const = False

if naive_reward == "True":
    naive_reward = True
elif naive_reward == "False":
    naive_reward = False


SelfPPO = TypeVar("SelfPPO", bound="PPO_MPC")


class PPO_MPC(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
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
        MSE_Weight: float = 1,
        Initial_Beta: float = 1,
        start_iteration_number: int = 0,
        end_iteration_number: int = 10
        
    )-> None:

        super(PPO_MPC, self).__init__(
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
            rollout_buffer_class = rollout_buffer_class,
            rollout_buffer_kwargs = rollout_buffer_kwargs,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.I_meas_history: list[np.ndarray] = []
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a multiple of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        # keep track of how many times we've done an update
        self._n_updates = 0


        # Additional MPC/imitation parameters:
        self.episode_length = 800
        self.adaptive_beta = Adaptive_Beta
        self.hu_weight = MSE_Weight
        self.initial_beta = Initial_Beta
        self.just_beta = Just_Beta
        self.sp = start_iteration_number/n_steps -1
        self.ep = end_iteration_number/n_steps - 1
        

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PPO_MPC, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)


        # For logging losses
        td_losses, imitation_losses, Main_losses = [], [], []

        # Count how many *completed* rollouts we have under our belt
        # (each rollout produces n_epochs of gradient‐steps)
        completed_rollouts = self._n_updates // self.n_epochs

        # 2) per‐epoch linear β schedule:
        if self.adaptive_beta:
            if completed_rollouts <= self.sp:
                # still in warm-up: β = initial
                self.beta = self.initial_beta
            elif completed_rollouts >= self.ep:
                # past end point: β = 0
                self.beta = 0.0
            else:
                # linearly go from initial→0 over [sp…ep]
                frac = (completed_rollouts - self.sp) / float(self.ep - self.sp)
                self.beta = self.initial_beta * (1.0 - frac)
        else:
            self.beta = self.initial_beta

        # print(f"adaptive_beta: {self.adaptive_beta}, beta: {self.beta}, completed_rollouts: {completed_rollouts}, sp: {self.sp}, ep: {self.ep}")

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            

            


            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):


                # --- unpack the new infos field ---
                obs_batch      = rollout_data.observations
                values_old     = rollout_data.old_values
                logp_old       = rollout_data.old_log_prob
                advantages     = rollout_data.advantages
                returns        = rollout_data.returns
                infos_batch    = rollout_data.infos   # <-- list of dicts

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())


                td_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                td_losses.append(td_loss.item())

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl and self.beta == 0.0:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # === new: extract expert actions and compute imitation loss ===
                expert_actions = th.tensor(
                    [info["Action MPC"] for info in infos_batch],
                    device=self.device,
                )
                if isinstance(self.action_space, spaces.Discrete):
                    # match SB3’s discrete-action format
                    expert_actions = expert_actions.long().flatten()


             
                # 1) Get the agent’s raw action logits
                dist = self.policy.get_distribution(obs_batch)
                logits = dist.distribution.logits              # shape [batch_size, n_actions]
                 
                # 2) Cross-entropy against the expert’s discrete action
                #    expert_actions is a LongTensor of shape [batch_size]
                imitation_loss = F.cross_entropy(logits, expert_actions.long().flatten())
                imitation_losses.append(imitation_loss.item())


                # Combine losses similar to the SAC code:
                if self.initial_beta == 100000:
                    loss = self.hu_weight * imitation_loss
                else:
                    if self.just_beta:
                        if self.beta == 1.0:
                            loss = (self.hu_weight * imitation_loss) + td_loss
                        elif self.beta == 0.0:
                            loss = td_loss
                        else:
                            loss = (self.hu_weight * self.beta * imitation_loss) + td_loss
                    else:
                        if self.beta == 1.0:
                            loss = self.hu_weight * imitation_loss
                        elif self.beta == 0.0:
                            loss = td_loss
                        else:
                            loss = (self.hu_weight * self.beta * imitation_loss) + (1 - self.beta) * td_loss
                            # loss = (self.hu_weight * self.beta * imitation_loss * td_loss) + (1 - self.beta) * td_loss 


                Main_losses.append(loss.item())

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        
        self._n_updates += self.n_epochs
                # ——— Plot I_meas trajectory in complex plane ———
        # ——— flatten the history into one big array ———
        all_obs = self.rollout_buffer.observations
        I_meas  = all_obs["I_meas"]            # shape = (n_steps*n_envs,2)
        I_meas_real = I_meas[:, 0]
        I_meas_imag = I_meas[:, 1]
        dists = np.sqrt(I_meas_real**2 + I_meas_imag**2)
        outside = dists > 30
        portion_outside = outside.sum() / len(dists)

        # Accumulate into your history array
        self.I_meas_history.append(I_meas.copy())

        full_I = np.concatenate(self.I_meas_history, axis=0)   # shape = (total_samples, 2)

        
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/portion_outside", portion_outside)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        self.logger.record("train/td_loss", np.mean(td_losses))
        self.logger.record("train/imitation_loss", np.mean(imitation_losses))
        self.logger.record("train/Main_loss", np.mean(Main_losses))
        self.logger.record("train/Beta", self.beta)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO_MPC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        # 1) run the standard learning, capturing the returned model instance
        model = super(PPO_MPC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

        # 2) now that training is done, dump I_meas
        
        # 1) stack every rollout’s array along the first axis
        full_I_meas = np.concatenate(self.I_meas_history, axis=0)   # shape = (total_samples, 2)

        # 2) make a DataFrame and write CSV
        df = pd.DataFrame(full_I_meas, columns=["real", "imag"])
        from pathlib import Path
        log_dir = Path(self.logger.get_dir())
        run_name  = log_dir.name
        weights_dir = log_dir / "Weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        csv_path = weights_dir / f"I_meas_full_buffer_{run_name}.csv"
        df.to_csv(csv_path, index=False)



        full_I_meas_real = full_I_meas[:,0]
        full_I_meas_img = full_I_meas[:,1]

        # ——— build a 0→1 “time” array over the full history ———
        N = full_I_meas_real.shape[0]
        time_idx = np.arange(N)
        colors   = time_idx / float(N - 1)

        # ——— scatter in the complex plane ———
        fig, ax = plt.subplots()
        sc = ax.scatter(full_I_meas_real,
                        full_I_meas_img,
                        c=colors,
                        cmap="viridis",
                        s=5,
                        alpha=0.8)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title(f"Measured Current")
        fig.colorbar(sc, ax=ax, label="progress through training")

        # ——— add a circle of radius 30 at the origin ———
        circle = Circle((0, 0), radius=30, edgecolor='red', fill=False, linewidth=2)
        ax.add_patch(circle)

        # ——— save to disk ———
        from pathlib import Path
        # Save plot into this run's folder
        log_dir = Path(self.logger.get_dir())
        run_name  = log_dir.name 
        plots_dir = log_dir / "Weights"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plots_dir / f"I_meas_train_{run_name}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        # ————————————————————————————————

        return model


def inject_clipped_variance(param,
                                var_ratio,
                                min_frac=1e-12,
                                max_frac=2.0):
        """
        param           : nominal positive value
        var_ratio       : noise variance relative to param^2 (e.g. 0.10 for “10% variance”)
        min_frac        : lower bound as a fraction of param (default → non‐negative)
        max_frac        : upper bound as a fraction of param (default → up to 200% of nominal)
        
        Returns:
        noisy_clipped : the parameter after Gaussian noise and clipping
        percent_error : (noisy_clipped - param) / param * 100
        """

        error = var_ratio * param

        noisy_clipped = param + error

        percent_error = round((error / param) * 100.0, 2)

        return noisy_clipped, percent_error

#########################################################################
def run_main(
    # seed & MPC / imitation settings
    Seed: int                 = 2778197655,
    MSE_Weight: float         = 1.0,

    # Environment & rollout
    env_name: str             = "VSC_Env-v0",
    n_steps: int              = 801,

    # PPO hyper‐parameters
    algo: str                 = 'PPO_MPC',
    batch_size: int           = 128,
    n_epochs: int             = 10,
    learning_rate: float      = 1e-4,
    gamma: float              = 0.98,
    gae_lambda: float         = 0.95,
    clip_range_type: str      = "fixed",
    clip_range: float         = 0.2,
    clip_range_vf: Optional[float] = None,
    ent_coef: float           = 0.0,
    vf_coef: float            = 0.5,
    max_grad_norm: float      = 0.5,

    # State‐dependent exploration (optional)
    use_sde: bool             = False,
    sde_sample_freq: int      = -1,

    # Early‐stop on KL divergence (optional)
    target_kl: Optional[float] = None,

    # Logging & evaluation
    tensorboard_log: Optional[str] = None,


    # Model / policy
    policy: Union[str, type[ActorCriticPolicy]] = "MultiInputPolicy",
    policy_kwargs: Optional[dict[str, Any]]       = None,

    # General settings
    verbose: int             = 1,
    total_timesteps: int     = 1_000_000,
    MPC_error_ratio: float   = 0.0,
    safety_const:  bool      = True,  
    naive_reward:  bool      = False,
    margin: float            = 0,
    device: Union[str, th.device] = "auto",

    # MPC-imitation scheduling
    Adaptive_Beta: bool      = False,
    Just_Beta: bool          = False,
    Initial_Beta: float      = 1.0,
    start_iteration_number: int = 1000,
    end_iteration_number: int   = 1_000_000,
    ) -> None:
        

    if clip_range_type == "fixed":
        clip_range = clip_range
    elif clip_range_type == "Inc":
        clip_range =  get_linear_fn(
        start=0.01,      # ε at progress=1.0
        end=0.2,       # ε at progress such that (1 - progress) >= end_fraction
        end_fraction=0.1
        )
    elif clip_range_type == "Dec":
        clip_range = get_linear_fn(
        start=0.2,      # ε at progress=1.0
        end=0.01,       # ε at progress such that (1 - progress) >= end_fraction
        end_fraction=0.1
        )
    
    Rf=0.013
    Lf=2.5e-3
    C=30e-6
    Rl=50
    
    # Example usage
    var_ratio = MPC_error_ratio            # “10% error” refers to variance = 0.1·param²
    Rf_MPC, Rf_MPC_percent_error = inject_clipped_variance(Rf,  var_ratio) 
    Lf_MPC, Lf_MPC_percent_error = inject_clipped_variance(Lf,  var_ratio) 
    C_MPC, C_MPC_percent_error  = inject_clipped_variance(C,   var_ratio) 
    Rl_MPC, Rl_MPC_percent_error = inject_clipped_variance(Rl,  var_ratio) 
    
    
    
    SEED = Seed
    # start_iteration_number = 0

    assert (start_iteration_number < end_iteration_number), (
        f"`start_iteration_number` must be less than `end_iteration_number`. "
        f"Currently start_iteration_number={start_iteration_number} and end_iteration_number={end_iteration_number}"
    )
    Save_results = './Results_' + envname + '/'

    import datetime

    # Create a timestamp string, e.g. 20250324_153045
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    net_arch = policy_kwargs.get("net_arch", None)
    arch_str = "-".join(str(x) for x in net_arch)
    policy_str = f"arch_{arch_str}"

    # Build a logname similar to the SAC version:
    if Adaptive_Beta:
        if Just_Beta:
            logname = (envname + '_' + algo + "_Safety_" + str(safety_const) + "_NRew_" + str(naive_reward) + "Marg" + str(margin) + '_W_' + str(MSE_Weight) +
                       '_Just_Beta_' + str(Initial_Beta) +
                       '_s_' + str(start_iteration_number) +
                       '_e_' + str(end_iteration_number) +
                       '_gma_' + str(gamma) +
                       '_TStp_' + str(total_timesteps) +
                       '_NStp_' + str(n_steps) +
                       '_NEpc_' + str(n_epochs) +
                       "_er_" + str(MPC_error_ratio) +
                       "_Sen_" + MPC_senario +
                       "_lr_" + str(learning_rate) +
                       "_kl_" + str(target_kl) +
                       "_entc_" + str(ent_coef) +
                       "_clpr_" + clip_range_type +
                       "_" + policy_str + 
                       "_sd_" + str(Seed) +
                       '_' + timestamp)
        else:
            logname = (envname + '_' + algo + "_Safety_" + str(safety_const) + "_NRew_" + str(naive_reward) + "Marg" + str(margin) + '_W_' + str(MSE_Weight) +
                       '_Beta_' + str(Initial_Beta) +
                       '_s_' + str(start_iteration_number) +
                       '_e_' + str(end_iteration_number) +
                       '_gma_' + str(gamma) +
                       '_TStp_' + str(total_timesteps) +
                       '_NStp_' + str(n_steps) +
                       '_NEpc_' + str(n_epochs) +
                       "_er_" + str(MPC_error_ratio) +
                       "_Sen_" + MPC_senario +
                       "_lr_" + str(learning_rate) +
                       "_kl_" + str(target_kl) +
                       "_entc_" + str(ent_coef) +
                       "_clpr_" + clip_range_type +
                       "_" + policy_str + 
                       "_sd_" + str(Seed) +
                        '_' + timestamp)
    else:
        if Just_Beta:
            logname = (envname + '_' + algo + "_Safety_" + str(safety_const) + "_NRew_" + str(naive_reward) + "Marg" + str(margin) + '_W_' + str(MSE_Weight) +
                       '_Just_Fx_Beta_' + str(Initial_Beta) +
                       '_s_' + str(start_iteration_number) +
                       '_e_' + str(end_iteration_number) +
                       '_gma_' + str(gamma) +
                       '_TStp_' + str(total_timesteps) +
                       '_NStp_' + str(n_steps) +
                       '_NEpc_' + str(n_epochs) +
                       "_er_" + str(MPC_error_ratio) +
                       "_Sen_" + MPC_senario +
                       "_lr_" + str(learning_rate) +
                       "_kl_" + str(target_kl) +
                       "_entc_" + str(ent_coef) +
                       "_clpr_" + clip_range_type +
                       "_" + policy_str + 
                       "_sd_" + str(Seed) +
                       '_' + timestamp)
        else:
            logname = (envname + '_' + algo + "_Safety_" + str(safety_const) + "_NRew_" + str(naive_reward) + "Marg" + str(margin) + '_W_' + str(MSE_Weight) +
                       '_Fx_Beta_' + str(Initial_Beta) +
                       '_s_' + str(start_iteration_number) +
                       '_e_' + str(end_iteration_number) +
                       '_gma_' + str(gamma) +
                       '_TStp_' + str(total_timesteps) +
                       '_NStp_' + str(n_steps) +
                       '_NEpc_' + str(n_epochs) +
                       "_er_" + str(MPC_error_ratio) +
                       "_Sen_" + MPC_senario +
                       "_lr_" + str(learning_rate) +
                       "_kl_" + str(target_kl) +
                       "_entc_" + str(ent_coef) +
                       "_clpr_" + clip_range_type +
                       "_" + policy_str + 
                       "_sd_" + str(Seed) +
                       '_' + timestamp)
    if Initial_Beta == 1000:
        logname = (envname + '_' + algo + "_Safety_" + str(safety_const) + "_NRew_" + str(naive_reward) + "Marg" + str(margin) + '_only_MSE_' +
                   '_W_' + str(MSE_Weight) +
                   '_s_' + str(start_iteration_number) +
                   '_e_' + str(end_iteration_number) +   
                   '_gma_' + str(gamma) +
                   '_TStp_' + str(total_timesteps) +
                   '_NStp_' + str(n_steps) +
                   '_NEpc_' + str(n_epochs) +
                   "_er-" + str(MPC_error_ratio) +
                   "_Sen_" + MPC_senario +
                   "_lr_" + str(learning_rate) +
                   "_kl_" + str(target_kl) +
                   "_entc_" + str(ent_coef) +
                   "_clpr_" + clip_range_type +
                   "_" + policy_str + 
                   "_sd_" + str(Seed) +
                   '_' + timestamp)
    if Initial_Beta == 0:
        logname = (envname + '_' + algo + "_Safety_" + str(safety_const) + "_NRew_" + str(naive_reward) + "Marg" + str(margin) + '_only_RL_' +
                   '_W_' + str(MSE_Weight) +
                   '_s_' + str(start_iteration_number) +
                   '_e_' + str(end_iteration_number) +   
                   '_gma_' + str(gamma) +
                   '_TStp_' + str(total_timesteps) +
                   '_NStp_' + str(n_steps) +
                   '_NEpc_' + str(n_epochs) +
                   "_er_" + str(MPC_error_ratio) +
                   "_Sen_" + MPC_senario +
                   "_lr_" + str(learning_rate) +
                   "_kl_" + str(target_kl) +
                   "_entc_" + str(ent_coef) +
                   "_clpr_" + clip_range_type +
                   "_" + policy_str + 
                   "_sd_" + str(Seed) +
                   '_' + timestamp)

    
    i = 1
    while os.path.isdir(Save_results + logname + '_' + str(i)):
        i += 1
    logname = logname + '_' + str(i)
    
    if SEED is None:
        np.random.seed(SEED)



    from stable_baselines3.common.monitor import Monitor



    raw_env = gymnasium.make(
        envname,
        MPC_error_ratio=MPC_error_ratio,
        senario=MPC_senario,
        action_discrete=True,
        Rf_MPC=Rf_MPC,
        Lf_MPC=Lf_MPC,
        C_MPC=C_MPC,
        Rload_MPC=Rl_MPC,
        safety_const = safety_const,
        Naive_reward = naive_reward,
        margin = margin
    )
    env = Monitor(raw_env,
    filename=None,  # or “./logs/monitor.csv” if you want the CSV on disk
    info_keywords=(
        "constraint violation",
        "constraint violation number",
    ),
    )

    os.makedirs(Save_results, exist_ok=True)
    new_logger = configure(Save_results + logname, ["stdout", "csv", "tensorboard"])
    # Assuming env is already created

    # Optionally, if your environment supports it (e.g. if it's a VecEnv):
    n_envs = env.num_envs if hasattr(env, "num_envs") else 1

    # Build the replay buffer kwargs with all parameters required by HerReplayBuffer
    # replay_buffer_kwargs = {
    #         "buffer_size": buffer_size,
    #         # "observation_space": env.observation_space,
    #         # "action_space": env.action_space,
    #         # "device": "auto",
    #         "n_envs": n_envs,
    #         "optimize_memory_usage": False,
    #         "handle_timeout_termination": True,
    #     }

    train_freq = n_steps
    print('Seed:', SEED, '| algo:', algo, '| envname:', envname, "| safety const:", safety_const, "| Naive_reward:", naive_reward, "| Margin: ", margin)
    print('Adaptive_Beta:', Adaptive_Beta, '| Just_Beta:', Just_Beta, '| Initial_Beta:', Initial_Beta, '| MSE_Weight:', MSE_Weight)
    print('total_timesteps:', total_timesteps, 
          '| start_iteration_number:', start_iteration_number, '| end_iteration_number:', end_iteration_number, "| MPC_error_ratio: ", MPC_error_ratio,
          '| gamma:', gamma, '| n_epochs:', n_epochs, '| n_steps:', n_steps)


    
    model = PPO_MPC(
    # Core PPO args
    policy=MultiInputPolicy,
    env=env,
    learning_rate=learning_rate,
    n_steps=n_steps,                # rollout length per env
    batch_size=batch_size,                 # minibatch size
    n_epochs=n_epochs,              # epochs per update
    gamma=gamma,                    # discount factor
    gae_lambda=gae_lambda, #0.95              # GAE lambda
    clip_range=clip_range,  #0.2               # PPO clip param
    clip_range_vf=clip_range_vf,             # no VF clipping
    normalize_advantage=True,
    ent_coef=ent_coef,     #0.0             # entropy coefficient
    vf_coef=vf_coef,                    # value function coefficient
    max_grad_norm=max_grad_norm,             # gradient clipping
    use_sde=use_sde,                  # state-dependent exploration?
    sde_sample_freq=sde_sample_freq,             # how often to resample SDE noise

    # Buffer / logging
    rollout_buffer_class=DictRolloutBuffer,
    rollout_buffer_kwargs=None,
    target_kl=target_kl,                 # no early stop on KL
    tensorboard_log=Save_results,

    # Policy & general
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
    device=device,
    _init_setup_model=True,

    # MPC-imitation parameters
    Adaptive_Beta=Adaptive_Beta,
    Just_Beta=Just_Beta,
    MSE_Weight=MSE_Weight,
    Initial_Beta=Initial_Beta,
    start_iteration_number=start_iteration_number,
    end_iteration_number=end_iteration_number,
    )

    model.set_logger(new_logger)
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback

    raw_env_eval = gymnasium.make(envname, MPC_error_ratio=MPC_error_ratio, senario=MPC_senario, action_discrete=True, Rf_MPC=Rf_MPC, Lf_MPC=Lf_MPC, C_MPC=C_MPC, Rload_MPC=Rl_MPC, safety_const=safety_const, Naive_reward=naive_reward, margin=margin)
    env_eval = Monitor(raw_env_eval,
    filename=None,  # or “./logs/monitor.csv” if you want the CSV on disk
    info_keywords=(
        "constraint violation",
        "constraint violation number",
    ),
    )

    eval_callback = EvalCallback(env_eval, best_model_save_path=Save_results + logname, verbose=1, warn=True,
                                    log_path=Save_results + logname, eval_freq=int(10*801),
                                    n_eval_episodes=10, deterministic=True, render=False)

    callback = CallbackList([eval_callback])

    model.learn(total_timesteps=total_timesteps, log_interval=2, tb_log_name=logname, callback=callback)
    os.makedirs(os.path.join(Save_results, "Weights"), exist_ok=True)
    model.save(Save_results + 'Weights/' + logname)

# policy_kwargs = dict(net_arch=[20, 20])
algo = 'PPO_MPC'
envname = 'VSC_Env-v0'
Save_results = './Results_' + envname + '/'
Adaptive_Beta      = True
Just_Beta       = False

# Example call using the previously parsed arguments:
run_main(
    # seed & MPC / imitation settings
    Seed=Seed,
    MSE_Weight=MSE_Weight,

    # Environment & rollout
    env_name="VSC_Env-v0",
    n_steps=n_steps,               # e.g. 800

    # PPO hyper‐parameters
    algo = algo,
    batch_size=128,
    n_epochs=400,
    learning_rate=learning_rate,
    gamma=gamma,                   # from your args
    gae_lambda=0.95,
    clip_range_type = clip_range_type,
    clip_range=0.2,
    clip_range_vf=None,
    ent_coef=ent_coef,
    vf_coef=0.5,
    max_grad_norm=10.0,

    # State‐dependent exploration (optional)
    use_sde=False,
    sde_sample_freq=-1,

    # Early‐stop on KL divergence (optional)
    target_kl=target_kl,   #None

    # Logging & evaluation
    tensorboard_log=Save_results,

    # Model / policy
    policy="MultiInputPolicy",
    policy_kwargs=policy_kwargs,

    # General settings
    verbose=1,
    total_timesteps=total_timesteps,
    MPC_error_ratio=MPC_error_ratio,
    safety_const = safety_const,
    naive_reward = naive_reward,
    margin = margin,
    device="cuda",

    # MPC‐imitation scheduling
    Adaptive_Beta=Adaptive_Beta,
    Just_Beta=Just_Beta,
    Initial_Beta=Initial_Beta,
    start_iteration_number=start_iteration_number,
    end_iteration_number=end_iteration_number,
)
_SIM.close()
