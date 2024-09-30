"""
This module contains concrete implementations for `MdpDecoder`s.
Implemented classes:
- MlpDecoder

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-22
"""

import torch
from typing import List, Type, Union, Tuple

from .mdpvae import MdpDecoder
from ..utility.distributions import DiagonalMultivariateNormal

class MlpDecoder(MdpDecoder):
    """A simple MDP-Decoder which uses a Multi-layer perceptron (MLP)
    to predict the means of the reward and the next observation.

    The last layer is simply a linear layer without activations.

    Parameters
    ----------
    latent_dim : int
        Dimension of latent representations.
    observation_dim : int
        Dimension of observations in the MDP.
    action_dim : int
        Dimension of the action space in the MDP.
    hidden_sizes : List[int]
        The hidden layer dimensions of the MLP.
    activation_function : Type[torch.nn.Module]
        Activation function of the hidden layers.
    std_rew : float, optional
        Standard deviation for reward distribution, by default 1e-2
    std_obs : float, optional
        Standard deviation for observation distribution, by default 1e-1
    """
    def __init__(
        self,
        latent_dim: int,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: List[int],
        activation_function: Type[torch.nn.Module] = torch.nn.ReLU,
        std_rew: float = 1e-1,
        std_obs: float = 1e-1,
    ) -> None:
        super().__init__(latent_dim, observation_dim, action_dim)
        # Instantiate the network
        self._mlp = torch.nn.Sequential()
        input_size = observation_dim + action_dim + latent_dim
        for hidden_size in hidden_sizes:
            self._mlp.extend([
                torch.nn.Linear(input_size, hidden_size),
                activation_function()
            ])
            input_size = hidden_size
        # Instantiate the output heads (one for rewards, one for observations)
        self._reward_out = torch.nn.Linear(input_size, 1)
        self._obs_out = torch.nn.Linear(input_size, observation_dim)

        self._std_rew = torch.tensor(std_rew)   # In theory, these values could be learned as well --> TODO
        self._std_obs = torch.tensor(std_obs)   # (either by having additional output heads or by making them parameters).

    def to(self, device: torch.device):
        super().to(device)
        self._std_rew.to(device)
        self._std_obs.to(device)

    def forward(
        self,
        z: torch.Tensor,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.distributions.Distribution, torch.distributions.Distribution]:
        # Concatenate latents, observations, and actions
        if z.ndim > observation.ndim:   # z has additional Monte Carlo sample dimension
            observation = observation.expand(z.shape[0],*observation.shape)
            action = action.expand(z.shape[0],*action.shape)
        x = torch.cat((observation, action, z), dim=-1)

        # Compute means and variances
        x = self._mlp(x)
        reward_mean = self._reward_out(x)
        observ_mean = self._obs_out(x)
        reward_var = self._std_rew.square() * torch.ones(torch.Size([1])).to(self.device)
        observ_var = self._std_obs.square() * torch.ones(torch.Size([self.observation_dim])).to(self.device)
        return DiagonalMultivariateNormal(reward_mean, reward_var), DiagonalMultivariateNormal(observ_mean, observ_var) 


class SeparateMlpDecoder(MdpDecoder):
    """A simple MDP-Decoder which uses TWO Multi-layer perceptron (MLP)
    to predict the means of the reward and the next observation, one
    MLP for each of them.

    The last layer is simply a linear layer without activations.

    Parameters
    ----------
    latent_dim : int
        Dimension of latent representations.
    observation_dim : int
        Dimension of observations in the MDP.
    action_dim : int
        Dimension of the action space in the MDP.
    hidden_sizes : List[int]
        The hidden layer dimensions of the MLP.
    activation_function : Type[torch.nn.Module]
        Activation function of the hidden layers.
    std_rew : float, optional
        Standard deviation for reward distribution, ignored if train_std = True.
        By default 1e-1
    std_obs : float, optional
        Standard deviation for observation distribution, ignored if train_std = True.
        By default 1e-1
    train_std : bool, optional
        Set to True to train variance / standard deviation of the decoder 
        distributions (instead of having fixed values ``std_rew``, ``std_obs``).
        By default False
    """
    def __init__(
        self,
        latent_dim: int,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: List[int],
        activation_function: Type[torch.nn.Module] = torch.nn.ReLU,
        std_rew: float = 1e-1,
        std_obs: float = 1e-1,
        train_std: bool = False,
    ) -> None:
        super().__init__(latent_dim, observation_dim, action_dim)

        # Instantiate the network for mean computation
        self._mlp_reward = torch.nn.Sequential()
        self._mlp_state = torch.nn.Sequential()
        input_size = observation_dim + action_dim + latent_dim
        for hidden_size in hidden_sizes:
            self._mlp_reward.extend([
                torch.nn.Linear(input_size, hidden_size),
                activation_function()
            ])
            self._mlp_state.extend([
                torch.nn.Linear(input_size, hidden_size),
                activation_function()
            ])
            input_size = hidden_size

        # Instantiate the output heads
        self._mean_rew = torch.nn.Linear(input_size, 1)
        self._mean_obs = torch.nn.Linear(input_size, observation_dim)

        # Standard deviations
        std_rew = torch.ones([1]) * std_rew
        std_obs = torch.ones([observation_dim]) * std_obs
        self._std_rew = torch.nn.Parameter(std_rew, requires_grad=train_std)
        self._std_obs = torch.nn.Parameter(std_obs, requires_grad=train_std)

    def forward(
        self,
        z: torch.Tensor,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.distributions.Distribution, torch.distributions.Distribution]:
        # Concatenate latents, observations, and actions
        if z.ndim > observation.ndim:   # z has additional Monte Carlo sample dimension
            observation = observation.expand(z.shape[0],*observation.shape)
            action = action.expand(z.shape[0],*action.shape)
        x = torch.cat((observation, action, z), dim=-1) 

        # Compute means and variances
        h_reward = self._mlp_reward(x)
        h_observ = self._mlp_state(x)
        reward_mean = self._mean_rew(h_reward)
        observ_mean = self._mean_obs(h_observ)

        reward_var = self._std_rew.square()
        observ_var = self._std_obs.square()
        
        return DiagonalMultivariateNormal(reward_mean, reward_var), DiagonalMultivariateNormal(observ_mean, observ_var) 