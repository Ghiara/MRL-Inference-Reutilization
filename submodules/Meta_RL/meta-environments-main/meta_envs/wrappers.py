import gym
from typing import Dict, Any, Tuple
import numpy as np

from .base import MultiTaskEnv, Task
import random
from smrl.utility.console_strings import print_to_terminal

class DomainRandomizer(object):
    """A wrapper for domain randomization to help with transfer to 
    - actions
    - observations
    - rewards

    Parameters
    ----------
    env : MetaEnv
        Original, mult-free environment
    mult_levels : Tuple[float, float, float]
        multiplier levels for
        1. actions,
        2. observations,
        3. rewards
    """
    def __init__(self, env: gym.Env, max_multiplier: float, min_multiplier: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.multiplier = random.random()*(max_multiplier-min_multiplier) + min_multiplier
        self.env = env

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        print_str = "Multiplier: " + str(self.multiplier)
        # print_to_terminal(print_str)
        action = action*self.multiplier
        obs, rew, done, truncated, info = self.env.step(action)
        obs = obs*self.multiplier
        rew = rew*self.multiplier
        return obs, rew, done, truncated, info

    def __getattr__(self, name: str):
        if name in ['action_noise', 'observation_noise', 'reward_noise', 'env']:
            raise AttributeError
        return getattr(self.env, name)

class NoisyEnv(object):
    """A wrapper for environments which adds white Gaussian noise to 
    - actions
    - observations
    - rewards

    Parameters
    ----------
    env : MetaEnv
        Original, noise-free environment
    noise_levels : Tuple[float, float, float]
        Noise levels for
        1. actions,
        2. observations,
        3. rewards
    """
    def __init__(self, env: gym.Env, noise_levels: Tuple[float,float,float], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_noise, self.observation_noise, self.reward_noise = noise_levels
        self.env = env

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action += np.random.randn(*action.shape) * self.action_noise
        obs, rew, done, truncated, info = self.env.step(action)
        obs += np.random.randn(*obs.shape) * self.observation_noise
        rew += np.random.randn() * self.reward_noise
        return obs, rew, done, truncated, info

    def __getattr__(self, name: str):
        if name in ['action_noise', 'observation_noise', 'reward_noise', 'env']:
            raise AttributeError
        return getattr(self.env, name)


class StateTaskWrapper(MultiTaskEnv):
    """A wrapper which constructs a multi-task environment from an arbitrary
    environment by randomly sampling target states as task.

    The target state is sampled from the observation space each time that ``sample_task()``
    is called.

    Parameters
    ----------
    MultiTaskEnv : _type_
        _description_
    """
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self._target_state = env.observation_space.sample()

    @property
    def unwrapped(self) -> gym.Env:
        return self.env.unwrapped

    @property
    def task(self) -> Task:
        return dict(
            target_state = self._target_state
        )

    @property
    def task_encoding_shape(self) ->Tuple[int, ...]:
        return self.env.observation_space.shape

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool, bool, dict]:
        obs, rew, term, trunc, info =  self.env.step(action)
        rew = - np.linalg.norm(obs - self._target_state)
        return obs, self._target_state, rew, term, trunc, info

    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, dict]:
        obs, info =  self.env.reset(*args, **kwargs)
        return obs, self._target_state, info
    
    def sample_task(self) -> Task:
        self.env.observation_space.seed()
        # Seeding is required here to ensure that copied environments in multi-
        # threaded rollouts still perform randomly (otherwise, the task would
        # be the same for all copies of this environment)
        self._target_state = self.env.observation_space.sample()
        return self.task

    def __getattr__(self, name: str):
        if name in ['env', '_target_state', 'task', 'task_encoding_shape', 'step', 'reset', 'sample_task', 'unwrapped']:
            raise AttributeError
        else:
            return getattr(self.env, name)
