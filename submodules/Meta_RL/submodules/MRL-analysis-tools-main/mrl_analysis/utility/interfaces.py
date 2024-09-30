"""
This module defines the interface classes
- MdpEncoder
- MetaEnv

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2022-12-14
"""

from abc import ABC
from typing import Union, Tuple, List, Any, Dict

import torch
import numpy as np
import gym

class MdpEncoder(torch.nn.Module, ABC):
    """An interface for MDP encoders which process context input, e.g.
    - observations
    - actions
    - rewards
    - next observations
    and output the (approximate) latent posterior q(z|x).

    Interface
    ---------
    - ``forward()``: Uses context data to output a distribution over 
        the latent space
    - ``get_encoding()``: Uses context data to output an encoding of the latent
        distribution
    - ``get_encodings()``: Same as ``get_encoding()`` but for batched data.

    Note: Some 'interface' functions are already implemented:
    - ``get_encoding()``
    - ``get_encodings()``

    Properties
    ----------
    - ``encoding_dim``: Dimension of latent encodings (may differ from dimension
        of latent space!)
    - ``latent_dim``: Dimension of the latent space
    - ``context_size``: Length of context sequence
    - ``encoding_mode``: Determines how encodings are generated from the latent
        distribution
    
    Parameters
    ----------
    encoding_dim : int
        Dimension of latent encodings (can be different from latent_dim, 
        e.g. if we use 'mean_var' as encoding_mode)
    latent_dim : int
        Dimension of latent representations.
    context_size : int
        Length of the context sequences used for encoding.
    encoding_mode : str, optional
        Determines how encodings from ``get_encoding()`` and ``get_encodings()``
        are generated from the latent posterior. Accepted values are: \n
        | ``'sample'`` | ``'mean'`` | ``'mode'`` | ``'mean_var'`` | \n
        See the documentation of these two functions for detailed information on
        the possible values.
        You can change the value later by setting the property ``encoding_mode``
        to any of the values above.
        By default 'sample'
    """

    @property
    def context_size(self) -> int:
        raise NotImplementedError
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        terminals: torch.Tensor,
    ) -> torch.distributions.Distribution:
        """Returns the posterior probability q(z|x) based on the 
        recent MDP history (t-h, t-h+1, ..., t-1) (a.k.a. 'context') where t is the current timestep.

        Parameters
        ----------
        observations : torch.Tensor
            History of observations, shape (batch_size, context_size, obs_dim)
        actions : torch.Tensor
            History of actions, shape (batch_size, context_size, action_dim)
        rewards : torch.Tensor
            History of rewards, shape (batch_size, context_size, 1)
        next_observations : torch.Tensor
            History of next observations, shape (batch_size, context_size, obs_dim)
        terminals : torch.Tensor
            History of terminal indicators (~ end of trajectory),
            shape (batch_size, context_size, 1)
        """
        raise NotImplementedError

    def get_encoding(
        self,
        observations: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        rewards: Union[np.ndarray, torch.Tensor],
        next_observations: Union[np.ndarray, torch.Tensor],
        terminals: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Returns a sample from the latent space based on context
        data. NOT BATCHED! (See `get_encodings()` for batched version.)

        The encodings are generated from the latent posterior based on the input
        property ``self.encoding_mode``:
        - ``'sample'``: Sample a variable from the latent distribution (random)
        - ``'mean'``: Take the mean of the latent distribution (deterministic)
        - ``'mode'``: Take the mode of the latent distribution (deterministic) 
                (May not be supported by all distribution types)
        - ``'mean_var'``: Concatenate mean and variance (deterministic, with uncertainty information)

        Parameters
        ----------
        observations : Union[np.ndarray, torch.Tensor]
            History of observations, shape (context_size, obs_dim)
        actions : Union[np.ndarray, torch.Tensor]
            History of actions, shape (context_size, action_dim)
        rewards : Union[np.ndarray, torch.Tensor]
            History of rewards, shape (context_size, 1)
        next_observations : Union[np.ndarray, torch.Tensor]
            History of next observations, shape (context_size, obs_dim)
        terminals : Union[np.ndarray, torch.Tensor]
            History of terminal indicators (~ end of trajectory),
            shape (context_size, 1)

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Sample z ~ q(z|x),
            Same type as input data.
        """
        raise NotImplementedError

    def get_encodings(
        self,
        observations: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        rewards: Union[np.ndarray, torch.Tensor],
        next_observations: Union[np.ndarray, torch.Tensor],
        terminals: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Returns an encoding based on context data, batched.
        
        The encodings are generated from the latent posterior based on the input
        property ``self.encoding_mode``:
        - ``'sample'``: Sample a variable from the latent distribution (random)
        - ``'mean'``: Take the mean of the latent distribution (deterministic)
        - ``'mode'``: Take the mode of the latent distribution (deterministic) 
                (May not be supported by all distribution types)
        - ``'mean_var'``: Concatenate mean and variance (deterministic, with uncertainty information)

        Parameters
        ----------
        observations : Union[np.ndarray, torch.Tensor]
            History of observations, shape (batch_size, context_size, obs_dim)
        actions : Union[np.ndarray, torch.Tensor]
            History of actions, shape (batch_size, context_size, action_dim)
        rewards : Union[np.ndarray, torch.Tensor]
            History of rewards, shape (batch_size, context_size, 1)
        next_observations : Union[np.ndarray, torch.Tensor]
            History of next observations, shape (batch_size, context_size, obs_dim)
        terminals : Union[np.ndarray, torch.Tensor]
            History of terminal indicators (~ end of trajectory),
            shape (batch_size, context_size, 1)

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Sample z ~ q(z|x),
            Same type as input data.
        """
        raise NotImplementedError

class MdpDecoder(torch.nn.Module, ABC):
    """A decoder for MDPs which takes as inputs

    - latent samples z
    - the latest observation o
    - the latest action a

    and outputs distributions for

    - the reward predictions r
    - next observation predictions o'

    Interface
    ---------
    - ``forward()``: Uses inputs (arbitrary shape) to output a distribution over 
        the latent space

    Note: Some 'interface' functions are already implemented:
    - ``get_encoding()``
    - ``get_encodings()``

    Properties
    ----------
    - ``observation_dim``: Dimension of environment observations
    - ``action_dim``: Dimension of agent actions
    - ``latent_dim``: Dimension of the latent space

    Parameters
    ----------
    latent_dim : int
        Dimension of latent representations.
    observation_dim : int
        Dimension of observations in the MDP.
    action_dim : int
        Dimension of the action space in the MDP.
    """
    
    def forward(self, z: torch.Tensor, observation: torch.Tensor, action: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.distributions.Distribution]:
        """Returns the likelihoods p(reward|z, observation, action) 
        and p(next observation|z, observation, action) based on the latent encoding z,
        and the current observation and action.

        The distribution type is specified within the VAE main class.

        Parameters
        ----------
        z : torch.Tensor
            Latent encoding
        observation : torch.Tensor
            Current observation
        action : torch.Tensor
            Current action

        Returns
        -------
        Tuple[torch.distributions.Distribution, torch.distributions.Distribution]
            Distribution over the rewards p(reward|z, observation, action)
            Distribution over the next observations p(next observation|z, observation, action),
        """
        raise NotImplementedError

class Policy(torch.nn.Module, ABC):
    """An generic policy.

    Parameters
    ----------
    action_dim : int
        Size of the action space
    """
    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.distributions.Distribution:
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
        raise NotImplementedError

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
        raise NotImplementedError

    def forward(self, obs: torch.Tensor, *inputs: torch.Tensor, **kwargs) -> torch.distributions.Distribution:
        raise NotImplementedError


class MetaRLPolicy(StandardPolicy, ABC):
    """A policy which bases its actions on 
    - observations and
    - task encodings.
    """

    def get_action(self, obs: np.ndarray, encoding: np.ndarray, mode: str = 'sample') -> Tuple[float, dict]:
        """Return action for a *single* observation (with a single encoding context).

        Parameters
        ----------
        obs : np.ndarray
            Observation, shape (observation_dim)
        encoding : np.ndarray
            Latent context (e.g. from encoder), shape (encoding_dim)
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
        raise NotImplementedError

    def get_actions(self, obs: np.ndarray, encodings: np.ndarray, mode: str = 'sample') -> np.ndarray:
        """Return actions for a *multiple* observations (each with their own encoding).

        Parameters
        ----------
        obs : np.ndarray
            Observations, shape(batch_size, observation_dim)
        encodings : np.ndarray
            Contexts (e.g. from encoder), shape (batch_size, encoding_dim)
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

    def forward(self, obs: torch.Tensor, encoding: torch.Tensor, *args, **kwargs) -> torch.distributions.Distribution:
        """This function represents the actual policy,
        i.e. it maps observations and latent contexts to
        a distribution over actions.

        Parameters
        ----------
        obs : torch.Tensor
            Observation / state, shape (batch_size, obs_dim) OR (obs_dim)
        encoding : torch.Tensor
            Latent context, possibly created by an oracle (multi-task learning)
            or an encoder (meta-learning), shape (batch_size, encoding_dim) OR (encoding_dim)

        Returns
        -------
        Distribution
            Distribution over actions.
        """
        raise NotImplementedError


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
        raise NotImplementedError

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
        raise NotImplementedError

    def forward(self, context: Dict[str, torch.Tensor], *args, **kwarg) -> torch.distributions.Distribution:
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


class MetaQFunction(torch.nn.Module, ABC):
    """A value function which accepts as input a combination of 
    - observation
    - action
    - latent encoding.

    Parameters
    ----------
    obs_dim : int
        Observation size
    act_dim : int
        Action size
    encoding_dim : int
        Encoding size
    """

    def forward(self, observation: torch.Tensor, action: torch.Tensor, encoding: torch.Tensor):
        """Returns the (estimated) value of a observation-action-latent tuple.

        Parameters
        ----------
        observation : torch.Tensor
            Observation, shape (batch_size, observation_dim)
        action : torch.Tensor
            Action, shape (batch_size, action_dim)
        encoding : torch.Tensor
            Latent encoding, shape (batch_size, encoding_dim)
        """
        raise NotImplementedError


class ContextQFunction(torch.nn.Module, ABC):
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


class MetaEnv(gym.Env, ABC):
    """Interface for meta-reinforcement learning environments with tasks.

    Interface
    ---------
    - ``task``: (Property) Returns a representation of the task
    - ``sample_task()``: Sample a task from the set of tasks

    Properties
    ----------
    - ``task``: The currently active task

    """

    @property
    def task(self) -> Any:
        """The current task description, e.g. a dictionary, a numpy array, ...
        """
        raise NotImplementedError

    def sample_task(self):
        """Set the current task to a random task from the set/distribution of tasks.
        """
        raise NotImplementedError

    def render(self, mode: str, width: int, height: int) -> np.ndarray:
        """ Render the environment 
        
        Parameters
        ----------
        mode : str
            Render mode, e.g. 'human', 'rgb_array'
        width : int
            Render width (in pixels)
        height : int
            Render height (in pixels)
        """
        raise NotImplementedError