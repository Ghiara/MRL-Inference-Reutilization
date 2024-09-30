"""
This file implements mappings from complicated environment contexts to more
simple environment contexts. They are important for encoder transfer.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-03-28
"""

import torch
from typing import Tuple

def map_toy1d_cont_to_disc(
    observations: torch.Tensor, 
    actions: torch.Tensor, 
    rewards: torch.Tensor, 
    next_observations: torch.Tensor, 
    terminals: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Maps transitions from the continuous, one-dimensional goal environment
    to the discrete, one-dimensional goal environment.
    """
    observations = observations[...,:1]
    next_observations = next_observations[...,:1]
    actions = next_observations - observations
    return observations, actions, rewards, next_observations, terminals


def map_cheetah_to_toy1d(
    observations: torch.Tensor, 
    actions: torch.Tensor, 
    rewards: torch.Tensor, 
    next_observations: torch.Tensor, 
    terminals: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Maps transitions from the cheetah environment
    to the discrete, one-dimensional goal environment.
    """
    observations = observations[...,-3:-2]
    next_observations = next_observations[...,-3:-2]
    actions = next_observations - observations
    return observations, actions, rewards, next_observations, terminals

def map_cheetah_to_limited_toy1d(
    observations: torch.Tensor, 
    actions: torch.Tensor, 
    rewards: torch.Tensor, 
    next_observations: torch.Tensor, 
    terminals: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Maps transitions from the cheetah environment
    to the discrete, one-dimensional goal environment.
    """
    observations = observations[...,:1] / 100.0
    next_observations = next_observations[...,:1] / 100.0
    rewards = rewards / 100.0
    actions = next_observations - observations
    return observations, actions, rewards, next_observations, terminals