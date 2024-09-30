"""
This module contains instantiations of the base class ``MetaQFunction``.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-26
"""

import torch
from typing import List, Type

from .base import MetaQFunction


class MlpValueFunction(MetaQFunction):
    """A meta Q-function which concatenates inputs and passes them through a MLP
    network.
    
    Parameters
    ----------
    obs_dim : int
        Observation dimension
    act_dim : int
        Action dimension
    encoding_dim : int
        Encoding dimension
    hidden_sizes : List[int],
        Sizes of the hidden layers in the network
    activation_layer : Type[torch.nn.Module], optional
        Activation layers for the hidden layers, by default torch.nn.ReLU
    """
    def __init__(
        self, 
        obs_dim: int, 
        act_dim: int, 
        encoding_dim: int, 
        hidden_sizes: List[int], 
        activation_layer: Type[torch.nn.Module] = torch.nn.ReLU,
    ) -> None:
        super().__init__(obs_dim, act_dim, encoding_dim)

        sizes = [obs_dim + act_dim + encoding_dim] + list(hidden_sizes)
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.Linear(in_size, out_size))
            layers.append(activation_layer())
        layers.append(torch.nn.Linear(out_size, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor, action: torch.Tensor, encoding: torch.Tensor):
        x = torch.concatenate([observation, action, encoding], dim=-1)
        return self.network(x)