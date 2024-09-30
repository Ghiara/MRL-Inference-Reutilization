""" 
This module contains implementations for MDP-Encoders:
- ``GRUEncoder``
- ``MlpEncoder``
- ``AttentionEncoder``
See also: mdpvae.py

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-24
"""


import torch
import numpy as np
from typing import Union, Tuple, List, Type

from smrl.utility.distributions import DiagonalMultivariateNormal
from ..mdpvae import MdpEncoder
from .util import batched, pad_sequence_length


class MlpEncoder(MdpEncoder):
    """An encoder which uses a multi-layer perceptron to process MDP context data
    of fixed length (use zero padding if necessary).
    The data is simply stacked on top of each other to form inputs to the network.

    Inputs have shape (batch_size, context_size, *).

    Parameters
    ----------
    observation_dim : int
        Dimension of observations in the MDP.
    action_dim : int
        Dimension of actions.
    latent_dim : int
        Dimension of latent representations.
    encoding_dim : int
        Dimension of latent encodings (can be different from latent_dim, 
        e.g. if we use 'mean_var' as encoding_mode)
    context_size : int
        Length of the context sequences.
    hidden_sizes : int
        Hidden dimension of the MLP network.
    activation_function : Type[torch.nn.Module], optional
        Activation function for hidden layers, by default torch.nn.ReLU
    """
    def __init__(
        self, 
        observation_dim: int,
        action_dim: int,
        latent_dim: int,
        encoding_dim: int,
        context_size: int, 
        hidden_sizes: List[int],
        activation_function: Type[torch.nn.Module] = torch.nn.ReLU,
        *args,
        **kwargs
    ) -> None:
        super().__init__(observation_dim, action_dim, latent_dim, encoding_dim, context_size, *args, **kwargs)

        # Instantiate the network
        self.hidden_sizes = hidden_sizes
        self._mlp = torch.nn.Sequential()
        input_size = (2 * observation_dim + action_dim + 2) * self.context_size
        for hidden_size in hidden_sizes:
            self._mlp.extend([
                torch.nn.Linear(input_size, hidden_size),
                activation_function()
            ])
            input_size = hidden_size
        self._mean_layer = torch.nn.Linear(hidden_size, latent_dim) # Outputs the mean vector of the distribution
        self._var_layer = torch.nn.Linear(hidden_size, latent_dim)  # Outputs the variance vector of the distribution

    @batched
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        terminals: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:
    
        x = torch.cat((observations, actions, rewards, next_observations, terminals), dim=-1)
        batch_size = x.shape[0] if x.shape[0] != 0 else 1

        # If context is empty, return default distribution
        if x.nelement() == 0:
            mean = torch.zeros(torch.Size([batch_size, self.latent_dim])).to(self.device)
            var = torch.ones(torch.Size([batch_size, self.latent_dim])).to(self.device)
            return DiagonalMultivariateNormal(mean, var)

        x = pad_sequence_length(self.context_size, x)

        # Pass input sequence through MLP
        h = self._mlp(x.reshape(batch_size, -1))

        # Apply output layers to obtain distribution parameters
        mean = self._mean_layer(h)
        var = self._var_layer(h).exp()
        return DiagonalMultivariateNormal(mean, var)


class PairAggregationEncoder(MlpEncoder):
    """An encoder which uses a multi-layer perceptron to process MDP context data
    of variable length by comparing pairs (or tuples) of transitions. 
    Pairs are matched randomly.

    Inputs have shape (batch_size, context_size, *).

    Parameters
    ----------
    observation_dim : int
        Dimension of observations in the MDP.
    action_dim : int
        Dimension of actions.
    latent_dim : int
        Dimension of latent representations.
    encoding_dim : int
        Dimension of latent encodings (can be different from latent_dim, 
        e.g. if we use 'mean_var' as encoding_mode)
    context_size : int
        Length of the context sequences.
    hidden_sizes : int
        Hidden dimension of the MLP network.
    activation_function : Type[torch.nn.Module], optional
        Activation function for hidden layers, by default torch.nn.ReLU
    
    References
    ----------
    Garnelo, M. et al. (2018) ‘Neural Processes’, 
        in ICML Workshop on Theoretical Foundations and Applications of Deep 
        Generative Models. Available at: https://arxiv.org/abs/1807.01622
    """
    supports_variable_sequence_length = True

    def __init__(
            self, 
            observation_dim: int, 
            action_dim: int, 
            latent_dim: int, 
            encoding_dim: int, 
            context_size: int, 
            hidden_sizes: List[int], 
            activation_function: Type[torch.nn.Module] = torch.nn.ReLU, 
            pair_size: int = 2,
            *args, **kwargs
        ) -> None:
        super().__init__(
            observation_dim, 
            action_dim, 
            latent_dim, 
            encoding_dim, 
            context_size=pair_size, 
            hidden_sizes=hidden_sizes, 
            activation_function=activation_function, 
            *args, **kwargs
        )
        self.pair_size = pair_size
        self.context_size = context_size

    @batched
    def forward(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, terminals: torch.Tensor) -> Union[Tuple[torch.Tensor], torch.Tensor]:

        x = torch.cat((observations, actions, rewards, next_observations, terminals), dim=-1)
        batch_size = x.shape[0] if x.shape[0] != 0 else 1

        # If context is empty, return default distribution
        if x.nelement() == 0:
            mean = torch.zeros(torch.Size([batch_size, self.latent_dim])).to(self.device)
            var = torch.ones(torch.Size([batch_size, self.latent_dim])).to(self.device)
            return DiagonalMultivariateNormal(mean, var)

        batch_size, context_size = x.shape[0], x.shape[1]

        # Construct pairs of context samples
        pairs = np.random.choice(context_size, size=(context_size, self.pair_size - 1), replace=True)
        pairs = np.concatenate([np.arange(context_size)[:,None], pairs], axis=-1)
        x_ = torch.zeros([context_size, batch_size, self.pair_size, x.shape[-1]], device=self.device)
        for i, pair in enumerate(pairs):
            x_[i] = x[:, pair, :]

        # Pass input sequence through MLP
        h = self._mlp(x_.reshape(context_size, batch_size, -1))
        h = torch.mean(h, dim=0)

        # Apply output layers to obtain distribution parameters
        mean = self._mean_layer(h)
        var = self._var_layer(h).exp()
        return DiagonalMultivariateNormal(mean, var)

    