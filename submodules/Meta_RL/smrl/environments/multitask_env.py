"""
This module contains the class ``MultiTaskEnv``.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    <2023-02-09
"""


from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod
import gym
import numpy as np


Task = Dict[str, Any]


class MultiTaskEnv(gym.Env, ABC):
    """Abstract base class for multi-task environments.
    """

    @property
    @abstractmethod
    def task_encoding_shape(self) -> Tuple[int, ...]:
        """The size of the task encodings returned by ``step()`` and ``reset()``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def task(self) -> Task:
        """The current task as a dictionary. 
        
        This might not be identical to the task encodings returned by 
        ``step()`` and ``reset()`` but can be understood as a human-readable
        task description.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_task(self) -> Task:
        """Set the current task to a random task from the set/distribution of tasks.

        Returns
        -------
        Task
            The new task
        """
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool, bool, dict]:
        """Environment step of the multi-task environment. Additonally to the
        observation, this environment also returns a task encoding in form
        of a numpy array.

        Parameters
        ----------
        action : np.ndarray
            Action

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, bool, bool, dict]
            Observation,
            Task encoding,
            Reward,
            Terminal indicator,
            Truncation indicator,
            Environment information
        """
        return super().step(action)
    
    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Resets the environment. Additionally to the current observation,
        this environment also returns a task encoding in form of a numpy array.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, dict]
            Observation
            Task encoding
        """
        raise NotImplementedError