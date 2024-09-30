"""
This module contains the decorator classes for input/output modification of
encoders.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-21
"""

import torch
from typing import List, Callable, Tuple
import difflib

from ..mdpvae import MdpEncoder
from .base_decorator import EncoderDecorator


class InputOutputDecorator(EncoderDecorator):
    """
    An encoder decorator which allows to modify the inputs and / or outputs
    to / of ``forward()``.

    Parameters
    ----------
    input_map : Callable
        Maps encoder inputs (to ``forward()``) to modified encoder inputs before
        passing the arguments to the wrapped encoder.
        By default None
    output_map : Callable
        Maps the output distribution of the wrapped encoder to a modified 
        distribution. By default None
    """
    def __init__(
            self, 
            encoder: MdpEncoder,
            input_map: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], \
                                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] \
                                = None,
            output_map: Callable[[torch.distributions.Distribution], torch.distributions.Distribution] = None,
            *args, **kwargs) -> None:
        super().__init__(encoder, *args, **kwargs)

        self.input_map = input_map
        self.output_map = output_map

    def forward(
            self, 
            observations: torch.Tensor, 
            actions: torch.Tensor, 
            rewards: torch.Tensor, 
            next_observations: torch.Tensor, 
            terminals: torch.Tensor
        ) -> torch.distributions.Distribution:

        if self.input_map is not None:
            observations, actions, rewards, next_observations, terminals = self.input_map(
                observations, actions, rewards, next_observations, terminals
            )

        dist = self._wrapped_encoder.forward(observations, actions, rewards, next_observations, terminals)

        if self.output_map is not None:
            dist = self.output_map(dist)

        return dist


class InputMaskedEncoder(InputOutputDecorator):
    """An encoder decorator which masks specific inputs to the forward function.

    Parameters
    ----------
    encoder : MdpEncoder
        Encoder object to be wrapped
    mask : List[int]
        A list of arguments which signals the inputs that will be masked, can be
        any subset of 
        ``'observations'`` | ``'actions'`` | ``'rewards'`` | ``'next_observations'`` | ``'terminals'``
    *args, **kwargs
        Any other argument which is forwarded to MdpEncoder (the parent class
        of EncoderDecorator)
    """
    mask_list = ['observations', 'actions', 'rewards', 'next_observations', 'terminals']

    def __init__(self, encoder: MdpEncoder, mask: List[str], *args, **kwargs) -> None:

        # Input parsing
        for mask_string in mask:
            if not mask_string in self.mask_list:
                message = f"Value \"{mask_string}\" is not applicable for mask."
                closest_match = difflib.get_close_matches(mask_string, self.mask_list, n=1)
                if len(closest_match) != 0:
                    closest_match = closest_match[0]
                    message += f" Maybe you meant \"{closest_match}\"?"
                raise ValueError(message)
            
        def mask_input(observations, actions, rewards, next_observations, terminals):
            """
            This function masks inputs (i.e. sets them to zeros) based on the mask.
            """
            if 'observations' in mask:
                observations = torch.zeros_like(observations)
            if 'actions' in mask:
                actions = torch.zeros_like(actions)
            if 'rewards' in mask:
                rewards = torch.zeros_like(rewards)
            if 'next_observations' in mask:
                next_observations = torch.zeros_like(next_observations)
            if 'terminals' in mask:
                terminals = torch.zeros_like(terminals)
            return self._wrapped_encoder.forward(observations, actions, rewards, next_observations, terminals)
        
        super().__init__(encoder, input_map=mask_input, *args, **kwargs)



