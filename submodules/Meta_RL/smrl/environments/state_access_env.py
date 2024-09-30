"""
This module contains environment interface classes which define direct read & 
write access to the state variable.

This access can be useful to train a network (e.g. encoder or policy) when it
is necessary to have control over the exploration data.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-08
"""



from abc import ABC, abstractmethod
import numpy as np

from .meta_env import MetaEnv


class StateAccessEnv(ABC):
    """An environment interface which ensures that the environment has a property
    ``state`` which can be accessed. In particular, this property can be set!

    The environment's ``step()``-function should modify ``state`` and ``state``
    should give a full description of the current state of the environment.

    Attributes
    ----------
    state : np.ndarray
        The state of the environment
    observation : np.ndarray
        The observation of the current state of the environment
    """
    @property
    @abstractmethod
    def state(self) -> np.ndarray:
        """The environment's state.
        """
        raise NotImplementedError

    @state.setter
    @abstractmethod
    def state(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def observation(self) -> np.ndarray:
        """An observation of the current state of the environment. 
        This attribute cannot be set.
        """
        raise NotImplementedError

    
class StateAccessMetaEnv(MetaEnv, StateAccessEnv):
    """An environment interface which ensures that the meta environment has a property
    ``state`` which can be accessed. In particular, this property can be set!

    The environment's ``step()``-function should modify ``state`` and ``state``
    should give a full description of the current state of the environment.

    Attributes
    ----------
    state : np.ndarray
        The state of the environment
    """
    pass
