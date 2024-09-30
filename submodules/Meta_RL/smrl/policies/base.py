"""
This module contains the following base classes for policies and value functions:
- ``Policy``
- ``StandardPolicy``
- ``MetaRLPolicy``
- ``ContextPolicy``
- ``MetaQFunction``
- ``ContextQFunction``

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-23
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Union, Any

from rlkit.torch.distributions import Distribution
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy

from smrl.utility.ops import np_batch_to_tensor_batch


class Policy(torch.nn.Module, ABC):
    """An generic policy.

    Parameters
    ----------
    action_dim : int
        Size of the action space
    """
    def __init__(self, action_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device('cpu')
        self.action_dim = action_dim

    def to(self, device):
        super().to(device)
        self.device = device

    def reset(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, *inputs: torch.Tensor, **kwargs) -> Distribution:
        """This function represents the actual policy, i.e. it maps inputs to a 
        distribution over actions.

        This function is called when using `policy.get_action`. 

        Parameters
        ----------
        inputs: torch.Tensor
            Policy inputs

        Returns
        -------
        Distribution
            Distribution over actions.
        """
        raise NotImplementedError

    @abstractmethod
    def get_action(self, *inputs: Union[np.ndarray, Any], mode: str = 'sample', **kwargs) -> Tuple[np.ndarray, Dict]:
        """Get action for a single observation.

        Parameters
        ----------
        inputs : np.ndarray(s) | Any
            Policy inputs, unbatched
        mode : str, optional
            Determines how the action sample is generated from the policy distribution.
            Available options are:
            ``'sample'``: Random sample from the distribution
            ``'mode'``: Use the distribution mode
            ``'mean'``: Use the distribution mean,
            by default ``'sample'``

        Returns
        -------
        np.ndarray, dict
            Action sampled from the action space according to the policy distribution, shape (action_dim)
            Debugging dictionary
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_actions(self, *inputs: Union[np.ndarray, Any], mode: str = 'sample', **kwargs) -> np.ndarray:
        """Return actions for a *batched* inputs.

        Parameters
        ----------
        inputs : np.ndarray(s) | Any
            Policy inputs, batched
        mode : str, optional
            Determines how the action samples are generated from the policy distribution.
            Available options are:
            ``'sample'``: Random sample from the distribution
            ``'mode'``: Use the distribution mode
            ``'mean'``: Use the distribution mean,
            by default ``'sample'``

        Returns
        -------
        np.ndarray
            Actions sampled from the action space according to the policy distribution.
            Shape (batch_size, action_dim)
        """
        raise NotImplementedError

    def _actions_from_distribution(self, dist: Distribution, mode: str):
        """Generate actions from the action distribution."""
        if mode == 'sample':
            actions = dist.sample()
        elif mode == 'mean':
            actions = dist.mean
        elif mode == 'mode':
            actions = dist.mode
        else:
            raise ValueError('Unknown option for argument \'mode\'.')
        return actions


class StandardPolicy(Policy, ABC):
    """A standard policy which takes observations and maps them to actions / 
    distributions over actions.

    Parameters
    ----------
    obs_dim : int
        Size of the observation space
    action_dim : int
        Size of the action space
    """
    def __init__(self, obs_dim: int, action_dim: int, *args, **kwargs) -> None:
        super().__init__(action_dim, *args, **kwargs)
        self.obs_dim = obs_dim

    def get_action(self, obs: np.ndarray, *inputs: np.ndarray, mode: str = 'sample', **kwargs) -> Tuple[float, dict]:
        """Return action for a *single* observation (with a single encoding context).

        Parameters
        ----------
        obs : np.ndarray
            Observation, shape (observation_dim)
        inputs : np.ndarray(s)
            Other observation-like inputs, shape (*)
        mode : str
            Determines how the action sample is generated from the policy distribution.
            Available options are:
            ``'sample'``: Random sample from the distribution
            ``'mode'``: Use the distribution mode
            ``'mean'``: Use the distribution mean,
            by default ``'sample'``

        Returns
        -------
        np.ndarray, dict
            Action sampled from the action space according to the policy distribution, shape (action_dim)
            Debugging dictionary
        """
        actions = self.get_actions(obs[None], *[input[None] for input in inputs], mode = mode, **kwargs)
        return actions[0, :], {}

    def get_actions(self, obs: np.ndarray, *inputs: np.ndarray, mode: str = 'sample', **kwargs) -> np.ndarray:
        """Return actions for a *multiple* observations (each with their own encoding).

        Parameters
        ----------
        obs : np.ndarray
            Observations, shape(batch_size, observation_dim)
        inputs : np.ndarray(s)
            Other observation-like inputs, shape (batch_size, *)
        mode : str, optional
            Determines how the action samples are generated from the policy distribution.
            Available options are:
            ``'sample'``: Random sample from the distribution
            ``'mode'``: Use the distribution mode
            ``'mean'``: Use the distribution mean,
            by default ``'sample'``

        Returns
        -------
        np.ndarray
            Actions sampled from the action space according to the policy distribution.
            Shape (batch_size, action_dim)
        """
        torch_inputs = tuple(torch_ify(x) for x in inputs)
        dist = self.forward(torch_ify(obs), *torch_inputs, **kwargs)
        actions = self._actions_from_distribution(dist, mode=mode)
        return elem_or_tuple_to_numpy(actions)

    def forward(self, obs: torch.Tensor, *inputs: torch.Tensor, **kwargs) -> Distribution:
        raise NotImplementedError

class MultiTaskPolicy(StandardPolicy, ABC):
    """A policy which bases its actions on 
    - observations and
    - task encodings.

    The task encodings could be provided by the environment (multi-task RL), by
    an oracle, or by a trained context-encoder (meta-RL). For the latter case, 
    you can also use the class ``MetaRLPolicy`` to clarify how you intend to use
    the policy.

    Parameters
    ----------
    obs_dim : int
        Size of the observation space
    encoding_dim : int
        Size of the task encodings
    action_dim : int
        Size of the action space
    """

    def __init__(self, obs_dim: int, encoding_dim: int, action_dim: int, *args, **kwargs) -> None:
        super().__init__(obs_dim, action_dim, *args, **kwargs)
        self.encoding_dim = encoding_dim

    def get_action(self, obs: np.ndarray, encoding: np.ndarray, mode: str = 'sample') -> Tuple[float, dict]:
        """Return action for a *single* observation (with a single task encoding).

        Parameters
        ----------
        obs : np.ndarray
            Observation, shape (observation_dim)
        encoding : np.ndarray
            Task encoding (e.g. from environment or encoder), shape (encoding_dim)
        mode : str
            Determines how the action sample is generated from the policy distribution.
            Available options are:
            ``'sample'``: Random sample from the distribution
            ``'mode'``: Use the distribution mode
            ``'mean'``: Use the distribution mean,
            by default ``'sample'``

        Returns
        -------
        np.ndarray, dict
            Action sampled from the action space according to the policy distribution, shape (action_dim)
            Debugging dictionary
        """
        return super().get_action(obs, encoding, mode=mode)

    def get_actions(self, obs: np.ndarray, encodings: np.ndarray, mode: str = 'sample') -> np.ndarray:
        """Return actions for a *multiple* observations (each with their own encoding).

        Parameters
        ----------
        obs : np.ndarray
            Observations, shape(batch_size, observation_dim)
        encodings : np.ndarray
            Task encodings (e.g. from environment or encoder), shape (batch_size, encoding_dim)
        mode : str, optional
            Determines how the action samples are generated from the policy distribution.
            Available options are:
            ``'sample'``: Random sample from the distribution
            ``'mode'``: Use the distribution mode
            ``'mean'``: Use the distribution mean,
            by default ``'sample'``

        Returns
        -------
        np.ndarray
            Actions sampled from the action space according to the policy distribution.
            Shape (batch_size, action_dim)
        """
        return super().get_actions(obs, encodings, mode=mode)

    @abstractmethod
    def forward(self, obs: torch.Tensor, encoding: torch.Tensor, *args, **kwargs) -> Distribution:
        """This function represents the actual policy,
        i.e. it maps observations and latent contexts to
        a distribution over actions.

        Parameters
        ----------
        obs : torch.Tensor
            Observation / state, shape (batch_size, obs_dim) OR (obs_dim)
        encoding : torch.Tensor
            Task encoding (e.g. from environment or encoder), 
            shape (batch_size, encoding_dim) OR (encoding_dim)

        Returns
        -------
        Distribution
            Distribution over actions.
        """
        raise NotImplementedError


class MetaRLPolicy(MultiTaskPolicy):
    """A policy which bases its actions on 
    - observations and
    - task encodings.

    NOTE: This type of policy is functionally equivalent to multi-task policies
    (see ``MultiTaskPolicy``).
    Due to its importance for the meta-RL algorithm, it is represented by its 
    own class.

    Parameters
    ----------
    obs_dim : int
        Size of the observation space
    encoding_dim : int
        Size of the task encodings
    action_dim : int
        Size of the action space
    """
    pass


class ContextPolicy(Policy, ABC):
    """A contextualized policy which bases its actions on context dictionaries.

    Parameters
    ----------
    context_size : int
        Sequence length of the contexts.
    obs_dim : int
        Dimension of the observation space.
    act_dim : int
        Dimension of the action space
    """

    def __init__(self, context_size: int, obs_dim: int, act_dim: int, *args, **kwargs) -> None:
        super().__init__(act_dim, *args, **kwargs)
        self.context_size = context_size
        self.obs_dim = obs_dim


    def get_action(self, context: Dict[str, np.ndarray], mode: str = 'sample') -> Tuple[float, dict]:
        """Return action for a *single* observation (with a single encoding context).

        Parameters
        ----------
        context : Dict[str, np.ndarray]
            Dictionary with context entries:
            ``'observations'``,
            ``'actions'``,
            ``'rewards'``,
            ``'next_observations'``,
            ``'terminals'``.
            Each entry has shape (context_size, *)
        mode : str
            Determines how the action sample is generated from the policy distribution.
            Available options are:
            ``'sample'``: Random sample from the distribution
            ``'mode'``: Use the distribution mode
            ``'mean'``: Use the distribution mean,
            by default ``'sample'``

        Returns
        -------
        np.ndarray, dict
            Action sampled from the action space according to the policy distribution, shape (action_dim)
            Debugging dictionary
        """
        for key, value in context.items():
            context[key] = value[None]
        actions = self.get_actions(context, mode=mode)
        return actions[0, :], {}

    def get_actions(self, context: Dict[str, np.ndarray], mode: str = 'sample') -> np.ndarray:
        """Return actions for a *multiple* observations (each with their own encoding).

        Parameters
        ----------
        context : Dict[str, np.ndarray]
            Dictionary with context entries:
            ``'observations'``, 
            ``'actions'``,
            ``'rewards'``,
            ``'next_observations'``,
            ``'terminals'``. 
            Each entry has shape (batch_size, context_size, *)
        mode : str, optional
            Determines how the action samples are generated from the policy distribution.
            Available options are:
            ``'sample'``: Random sample from the distribution
            ``'mode'``: Use the distribution mode
            ``'mean'``: Use the distribution mean,
            by default ``'sample'``

        Returns
        -------
        np.ndarray
            Actions sampled from the action space according to the policy distribution.
            Shape (batch_size, action_dim)
        """
        dist = self.forward(np_batch_to_tensor_batch(context))
        actions = self._actions_from_distribution(dist, mode=mode)
        return elem_or_tuple_to_numpy(actions)

    @abstractmethod
    def forward(self, context: Dict[str, torch.Tensor], *args, **kwarg) -> Distribution:
        """This function represents the actual policy,
        i.e. it maps observations and latent contexts to
        a distribution over actions.

        Parameters
        ----------
        context : Dict[str, torch.Tensor]
            Dictionary with context entries:
            ``'observations'``, 
            ``'actions'``,
            ``'rewards'``,
            ``'next_observations'``,
            ``'terminals'``. 
            Each entry has shape (batch_size, context_size, *)

        Returns
        -------
        Distribution
            Distribution over actions.
        """
        raise NotImplementedError


class TargetPolicy(StandardPolicy):
    """A standard policy which has the additional attribute ``target``.
    You can use this attribute to determine intrinsic rewards.

    Parameters
    ----------
    obs_dim : int
        Dimension of the observation space.
    act_dim : int
        Dimension of the action space
    target : int
    """
    def __init__(self, obs_dim: int, action_dim: int, target: np.ndarray, *args, **kwargs) -> None:
        super().__init__(obs_dim, action_dim, *args, **kwargs)
        self.target = target


class ValueFunction(torch.nn.Module, ABC):
    """
    A value function which accepts arbitrary inputs and returns a value estimate.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device('cpu')

    def to(self, device):
        super().to(device)
        self.device = device

    @abstractmethod
    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Returns the value estimate for the given inputs.

        Parameters
        ----------
        *inputs : torch.Tensor
            Inputs for the value function
        """
        raise NotImplementedError
    
class MultiTaskQFunction(ValueFunction):
    """A value function which accepts as input a combination of 
    - observation
    - action
    - task encoding.

    Parameters
    ----------
    obs_dim : int
        Observation size
    act_dim : int
        Action size
    encoding_dim : int
        Task encoding size
    """
    def __init__(self, obs_dim: int, act_dim: int, encoding_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.encoding_dim = encoding_dim

    @abstractmethod
    def forward(self, observation: torch.Tensor, action: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        """Returns the (estimated) value of a observation-action-latent tuple.

        Parameters
        ----------
        observation : torch.Tensor
            Observation, shape (batch_size, observation_dim)
        action : torch.Tensor
            Action, shape (batch_size, action_dim)
        encoding : torch.Tensor
            Task encoding, shape (batch_size, encoding_dim)
        """
        raise NotImplementedError
    
class MetaQFunction(MultiTaskQFunction):
    """A value function which accepts as input a combination of 
    - observation
    - action
    - task encoding.

    Parameters
    ----------
    obs_dim : int
        Observation size
    act_dim : int
        Action size
    encoding_dim : int
        Task encoding size
    """
    pass


class ContextQFunction(ValueFunction):
    """A value function which accepts contexts and actions as inputs.

    Parameters
    ----------
    context_size : int
        Sequence length of the contexts.
    obs_dim : int
        Dimension of the observation space.
    act_dim : int
        Dimension of the action space
    """
    def __init__(self, context_size: int, obs_dim: int, act_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context_size = context_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    @abstractmethod
    def forward(self, context: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """Returns the (estimated) value of a context.

        Parameters
        ----------
        context : Dict[str, torch.Tensor]
            Dictionary with context entries:
            ``'observations'``, 
            ``'actions'``,
            ``'rewards'``,
            ``'next_observations'``,
            ``'terminals'``. 
            Each entry has shape (batch_size, context_size, *)

        Returns
        -------
        torch.Tensor
            Value prediction of shape (batch_size, 1)
        """
        raise NotImplementedError