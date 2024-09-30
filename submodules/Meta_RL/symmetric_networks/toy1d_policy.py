"""
This module contains equivariant policies and value functions for 
Toy1D.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-02
"""

import copy
import torch
from typing import List

from symmetrizer.nn.modules import BasisLinear
from symmetrizer.ops import GroupRepresentations
from symmetrizer.groups import Group

from rlkit.torch.distributions import TanhNormal

from smrl.policies.base import MetaQFunction, MetaRLPolicy 

from .groups import *

class Toy1dQFunctionToPermutationGroup(Group):
    """
    Equivariance group for Toy1d Q-functions.
    For negated observation, encoding, and action, the group returns the 
    permuted output.
    """
    def __init__(self, observation_dim, encoding_dim, action_dim):
        self.parameters = range(2)

        self._input_transforms = [lambda x: x, lambda x: -x]
        self._output_transforms = permutation_repr

        self.repr_size_in = observation_dim + encoding_dim + action_dim
        self.repr_size_out = 2
    
    def _input_transformation(self, weights, idx: int):
        return self._input_transforms[idx](weights)

    def _output_transformation(self, weights, idx: int):
        return self._output_transforms[idx] @ weights

class Toy1dPolicyToPermutationGroup(Group):
    """
    Equivariance group for Toy1d Q-functions.
    For negated observation, encoding, and action, the group returns the 
    permuted output.
    """
    def __init__(self, observation_dim, encoding_dim):
        self.parameters = range(2)

        self._input_transforms = [lambda x: x, lambda x: -x]
        self._output_transforms = permutation_repr

        self.repr_size_in = observation_dim + encoding_dim
        self.repr_size_out = 2
    
    def _input_transformation(self, weights, idx: int):
        return self._input_transforms[idx](weights)

    def _output_transformation(self, weights, idx: int):
        return self._output_transforms[idx] @ weights


class Toy1dQFunction(MetaQFunction):
    """An negation-invariant Q-function for Toy1D.

    ``value(obs, enc, act) = value(-obs, -enc, -act)``

    Parameters
    ----------
    obs_dim : int
        Observation dimension
    act_dim : int
        Action dimension
    encoding_dim : int
        Encoding dimension
    hidden_sizes : List[int]
        Sizes of the hidden layers
    """
    def __init__(self, obs_dim: int, act_dim: int, encoding_dim: int, hidden_sizes: List[int], *args, **kwargs) -> None:
        super().__init__(obs_dim, act_dim, encoding_dim, *args, **kwargs)
        layers = []

        hidden_sizes = copy.deepcopy(hidden_sizes)   # Make sure that you do not override the config file...

        layers.extend([# Input layer
            BasisLinear(
                channels_in=1,
                channels_out=hidden_sizes[0], 
                group=Toy1dQFunctionToPermutationGroup(obs_dim, encoding_dim, act_dim),
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

        layers.append(BasisLinear(# Output layer
            channels_in=out_size,
            channels_out=1,
            group=PermutationToInvariantGroup(1),
        ))
        
        self.network = torch.nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor, action: torch.Tensor, encoding: torch.Tensor):
        x = torch.concatenate([observation, encoding, action], dim=-1).unsqueeze(1)
        return self.network.forward(x).squeeze(1)
    

class Toy1dPolicy(MetaRLPolicy):
    """An equivariant policy for Toy1d.

    ``action(obs, enc) = action(-obs, -enc)``

    Parameters
    ----------
    obs_dim : int
        Observation dimension
    encoding_dim : int
        Encoding dimension
    act_dim : int
        Action dimension
    hidden_sizes : List[int]
        Sizes of the hidden layers
    """
    def __init__(self, obs_dim: int, encoding_dim: int, action_dim: int, hidden_sizes: List[int], *args, **kwargs) -> None:
        super().__init__(obs_dim, encoding_dim, action_dim, *args, **kwargs)
        layers = []

        hidden_sizes = copy.deepcopy(hidden_sizes)   # Make sure that you do not override the config file...

        layers.extend([# Input layer
            BasisLinear(
                channels_in=1,
                channels_out=hidden_sizes[0], 
                group=Toy1dPolicyToPermutationGroup(obs_dim, encoding_dim),
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

        self.mean_layer = BasisLinear(# Output layer
            channels_in=out_size,
            channels_out=1,
            group=PermutationToNegationGroup(1),
        )
        self.std_layer = BasisLinear(# Output layer
            channels_in=out_size,
            channels_out=1,
            group=PermutationToInvariantGroup(1),
        )
        
        self.network = torch.nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, encoding: torch.Tensor, *args, **kwargs) -> torch.distributions.Distribution:
        x = torch.concatenate([obs, encoding], dim=-1).unsqueeze(1)
        h = self.network.forward(x)
        mean = self.mean_layer(h).squeeze(1)
        std = self.std_layer(h).squeeze(1).exp()
        return TanhNormal(mean, std)