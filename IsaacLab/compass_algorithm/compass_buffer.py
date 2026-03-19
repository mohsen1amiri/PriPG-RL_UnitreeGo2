import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

class COMPASSReplayBufferSamples:
    """
    A custom data class to hold the standard RL batch PLUS the expert actions.
    """
    def __init__(self, samples: ReplayBufferSamples, expert_actions: torch.Tensor, has_expert: torch.Tensor):
        self.observations = samples.observations
        self.actions = samples.actions
        self.next_observations = samples.next_observations
        self.dones = samples.dones
        self.rewards = samples.rewards
        # Custom COMPASS attributes
        self.expert_actions = expert_actions
        self.has_expert = has_expert

class COMPASSBuffer(ReplayBuffer):
    """
    Custom Replay Buffer for COMPASS.
    It intercepts the 'info' dictionary from the environment step to save the REAP actions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create extra storage arrays in memory for the expert data
        self.expert_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.has_expert = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        """
        Overrides the standard add() to extract 'expert_action' from the info dictionary.
        """
        # Grab the current index before the standard add() increments it
        idx = self.pos
        
        for i, info in enumerate(infos):
            # Check if our main training loop passed the REAP action into the info dict
            if "expert_action" in info:
                self.expert_actions[idx, i] = info["expert_action"]
                self.has_expert[idx, i] = 1.0  # Mark that we have valid expert data here
            else:
                self.expert_actions[idx, i] = np.zeros(self.action_dim)
                self.has_expert[idx, i] = 0.0  # Mark as invalid 
                
        # Call standard SB3 add function to store obs, action, reward, etc.
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> COMPASSReplayBufferSamples:
        """
        Retrieves standard samples and attaches the expert data to them for the COMPASS train() function.
        """
        # Get standard batch
        samples = super()._get_samples(batch_inds, env)
        
        # Get our custom expert batch and convert to PyTorch tensors
        expert_actions = self.to_torch(self.expert_actions[batch_inds, 0, :])
        has_expert = self.to_torch(self.has_expert[batch_inds, 0]).unsqueeze(1)
        
        # Wrap everything in our custom sample object
        return COMPASSReplayBufferSamples(samples, expert_actions, has_expert)