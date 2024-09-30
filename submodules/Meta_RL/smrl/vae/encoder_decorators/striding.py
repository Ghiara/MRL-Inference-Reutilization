"""
This module contains the class ``StridedEncoder``.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-26
"""

import torch
import numpy as np
from typing import Type

from smrl.utility.console_strings import warning

from ..mdpvae import MdpEncoder
from ..encoder_networks.util import batched

from .base_decorator import EncoderDecorator


class StridedEncoder(EncoderDecorator):
    """A decorator for encoders which passes only a subset of the sequence 
    timesteps to ``forward()`` of the wrapped encoder.

    Parameters
    ----------
    encoder : MdpEncoder
        Encoder object to be wrapped
    stride : int
        Step size between passed forward inputs.
    *args, **kwargs
        Any other argument which is forwarded to MdpEncoder (the parent class
        of EncoderDecorator)

    Attributes
    ----------
    unwrapped : MdpEncoder
        The wrapped encoder object
    """
    def __init__(self, encoder: MdpEncoder, stride: int, *args, **kwargs) -> None:
        super().__init__(encoder, *args, **kwargs)
        self.stride = stride
        if self._wrapped_encoder.context_size != self.context_size // self.stride:
            print(warning(f"The wrapped encoder has a context size of {self._wrapped_encoder.context_size}"
             + f" altough it will receive at most (context_size // stride) = {self.context_size // self.stride} samples in a row"
             + " due to striding. \nConsider setting the context size of the wrapped encoder manually by passing"
             + f" 'context_size': {self.context_size // self.stride} in the config's 'encoder_kwargs'"))
    
    @batched
    def forward(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, terminals: torch.Tensor) -> torch.distributions.Distribution:
        idx = np.flip(observations.shape[1] - np.arange(0, observations.shape[1], self.stride)) - 1
        observations = observations[:, idx, ...]
        actions = actions[:, idx, ...]
        rewards = rewards[:, idx, ...]
        next_observations = next_observations[:, idx, ...]
        terminals = terminals[:, idx, ...]
        return self._wrapped_encoder.forward(observations, actions, rewards, next_observations, terminals)
