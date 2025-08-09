""" DM Control to Gym Wrapper """

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


class DMControlGymWrapper(gym.Env):
    """
    A wrapper for DMControl environments to make them compatible with OpenAI Gym.
    This wrapper flattens the observation space and handles the action space
    appropriately for use with stable_baselines3.
    """

    def __init__(self, dm_env, seed=None):
        self.dm_env = dm_env
        self._validate_env_specs()
        self.observation_space = self._convert_observation_space(
            self.dm_env.observation_spec()
        )
        self.action_space = self._convert_action_space(self.dm_env.action_spec())
        self.seed = seed
        if self.seed is not None:
            self._set_seeds(self.seed)

    def _convert_action_space(self, spec):
        low = np.full(spec.shape, spec.minimum)
        high = np.full(spec.shape, spec.maximum)
        return spaces.Box(low=low, high=high, dtype=np.float64)

    def _convert_observation_space(self, spec):
        if isinstance(spec, dict):
            low = np.concatenate([np.full(s.shape, -10) for s in spec.values()])
            high = np.concatenate([np.full(s.shape, 10) for s in spec.values()])
        else:
            low = np.full(spec.shape, spec.minimum)
            high = np.full(spec.shape, spec.maximum)
        return spaces.Box(low=low, high=high, dtype=np.float64)

    def _flatten_obs(self, obs):
        return np.concatenate([v.ravel() for v in obs.values()]).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        """Reset the environment and return initial observation and info."""
        if seed is not None:
            self._set_seeds(seed)
        time_step = self.dm_env.reset()
        obs = self._flatten_obs(time_step.observation)
        return obs, {}  # gymnasium requires (obs, info)

    def step(self, action):
        time_step = self.dm_env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = time_step.reward or 0.0
        terminated = time_step.last()
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _validate_env_specs(self):
        """Validate that the environment has proper specs."""
        try:
            obs_spec = self.dm_env.observation_spec()
            action_spec = self.dm_env.action_spec()
            if obs_spec is None or action_spec is None:
                raise ValueError("Environment specs cannot be None")
        except Exception as e:
            raise ValueError(f"Failed to validate environment specs: {e}")

    def _set_seeds(self, seed):
        """Set seeds for reproducibility."""
        self.action_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def get_observation_keys(self):
        """Get the keys of the observation space if it's a dictionary."""
        spec = self.dm_env.observation_spec()
        if isinstance(spec, dict):
            return list(spec.keys())
        return None

    def get_observation_shapes(self):
        """Get the shapes of each observation component."""
        spec = self.dm_env.observation_spec()
        if isinstance(spec, dict):
            return {k: v.shape for k, v in spec.items()}
        return spec.shape

    def render(self, mode="human", **kwargs):
        """Render the environment."""
        pass

    def close(self):
        """Close the environment."""
        pass

    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self.dm_env
