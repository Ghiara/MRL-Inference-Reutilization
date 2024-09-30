"""
This module contains the an abstract base class for meta / multitask environments.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-02
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import gym

class MetaEnv(gym.Env, ABC):
    """Abstract base class for meta environments.
    """
    def __init__(self) -> None:
        self.meta_mode = 'train'
        super().__init__()

    def set_meta_mode(self, mode='train'):
        """Set the environment to 'train' or 'test'.

        This changes the set of tasks from which the current task can be sampled.
        There are separate sets of train tasks and test tasks. 

        Parameters
        ----------
        mode : str, optional
            The meta mode, can either be 'train' or 'test', by default 'train'
        """
        assert (mode in ['train', 'test']), "Argument `mode` can only be set to 'train' or 'test'."
        self.meta_mode = mode

    @property
    @abstractmethod
    def task(self) -> Dict[str, Any]:
        """The current task description as a dictionary.

        The dictionary must contain the key 'id' which identifies the task 
        (could be an integer or a string).
        """
        raise NotImplementedError

    @abstractmethod
    def sample_task(self):
        """Set the current task to a random task from the set/distribution of tasks.
        """
        raise NotImplementedError