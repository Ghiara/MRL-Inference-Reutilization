"""
This module contains encoder networks for equivariant Toy1D training.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-13
"""

import numpy as np
import torch
import copy
from typing import List, Callable

from symmetrizer.nn.modules import BasisLinear
from symmetrizer.ops import GroupRepresentations
from symmetrizer.groups import Group

from smrl.vae.mdpvae import MdpEncoder
from smrl.vae.encoder_networks.util import batched, pad_sequence_length, at_least_one_timestep
from smrl.utility.distributions import DiagonalMultivariateNormal

from .equivariant_gru import EquivariantGRU
from .groups import *

class Toy1dToPermutationGroup(Group):
    """
    Equivariance group from Toy1d symmetry (negated observations)
    to 2d permutation transformations.
    """
    def __init__(self, observation_dim):
        self.parameters = range(2)

        self._input_transforms = GroupRepresentations(
            [
                np.eye(observation_dim + 1),
                np.diag(np.concatenate([-np.ones((observation_dim)), np.array([1])], axis=0))
            ],
            name="Negative Observations"
        )
        self._output_transforms = GroupRepresentations(
            [
                np.eye(2),
                np.array([[0, 1], [1, 0]])
            ],
            name = "Permutation",
        )

        self.repr_size_in = observation_dim + 1
        self.repr_size_out = 2
    
    def _input_transformation(self, weights, idx: int):
        return weights @ self._input_transforms[idx]

    def _output_transformation(self, weights, idx: int):
        return self._output_transforms[idx] @ weights


class Toy1dEncoder(MdpEncoder):
    """
    A symmetry-equivariant MLP encoder for Toy1d environments.

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
    hidden_sizes : List[int]
        Channel sizes in the hidden layers
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

    def __init__(self, observation_dim: int, action_dim: int, latent_dim: int, encoding_dim: int, hidden_sizes: List[int], context_size: int = None, encoding_mode: str = 'sample', *args, **kwargs) -> None:
        super().__init__(1, 1, latent_dim, encoding_dim, context_size, encoding_mode, *args, **kwargs)

        layers = []

        hidden_sizes = copy.deepcopy(hidden_sizes)   # Make sure that you do not override the config file...

        layers.extend([# Input layer
            BasisLinear(
                channels_in=context_size,
                channels_out=hidden_sizes[0], 
                group=Toy1dToPermutationGroup(observation_dim),
            ),
            torch.nn.Tanh(),
        ])

        for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.extend([# Hidden layers
                BasisLinear(
                    channels_in=in_size,
                    channels_out=out_size,
                    group=PermutationGroup(),
                ),
                torch.nn.Tanh(),
            ])

        self.network = torch.nn.Sequential(*layers)

        self.mean_layer = BasisLinear(
            channels_in=out_size,
            channels_out=1,
            group=PermutationToNegationGroup(latent_dim),
        )
        self.var_layer = BasisLinear(
            channels_in=out_size,
            channels_out=1,
            group=PermutationToInvariantGroup(latent_dim),
        )

    @batched
    def forward(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, terminals: torch.Tensor) -> torch.distributions.Distribution:
        x = torch.cat([observations, rewards], dim=-1)
        batch_size = x.shape[0] if x.shape[0] != 0 else 1

        # If context is empty, return default distribution
        if x.nelement() == 0:
            mean = torch.zeros(torch.Size([batch_size, self.latent_dim])).to(self.device)
            var = torch.ones(torch.Size([batch_size, self.latent_dim])).to(self.device)
            return DiagonalMultivariateNormal(mean, var)

        x = pad_sequence_length(self.context_size, x)
        
        # Pass input through the network
        x = self.network(x)
        mean = self.mean_layer(x)
        var = self.var_layer(x).exp()
        mean, var = mean.squeeze(1), var.squeeze(1)   # Remove channel dimension

        return DiagonalMultivariateNormal(mean, var)
        

class Toy1dGRUEncoder(MdpEncoder):
    """
    A symmetry-equivariant GRU encoder for Toy1d environments.

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
    hidden_sizes : int, optional
        Channel size in the hidden layers, by default 16
    """
    supports_variable_sequence_length = True
    def __init__(
        self, 
        observation_dim: int, 
        action_dim: int, 
        latent_dim: int, 
        encoding_dim: int, 
        context_size: int = None, 
        encoding_mode: str = 'sample', 
        hidden_size: int = 16,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(observation_dim, action_dim, latent_dim, encoding_dim, context_size, encoding_mode, *args, **kwargs)
        self.gru = EquivariantGRU(
            input_group = Toy1dToPermutationGroup(observation_dim),
            hidden_group = PermutationGroup(),
            channels_out = hidden_size,
            batch_first = True,
        )
        self.mean_layer = BasisLinear(
            channels_in=hidden_size,
            channels_out=1,
            group=PermutationToNegationGroup(latent_dim),
        )
        self.var_layer = BasisLinear(
            channels_in=hidden_size,
            channels_out=1,
            group=PermutationToInvariantGroup(latent_dim),
        )
        self.hidden_size = hidden_size

    @batched
    def forward(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, terminals: torch.Tensor) -> torch.distributions.Distribution:
        x = torch.cat([observations, rewards], dim=-1)
        _, h = self.gru.forward(x.unsqueeze(-2))    # Unsqueeze channel dimension...
        mean: torch.Tensor = self.mean_layer(h)
        var: torch.Tensor = self.var_layer(h).exp()
        mean, var = mean.squeeze(-2), var.squeeze(-2)  # Remove channel dimension

        return DiagonalMultivariateNormal(mean, var)
        