"""
This module contains the class ``MultiDecorator``

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-19
"""

from typing import Tuple, Type, Dict, List, Any
import torch

from ..mdpvae import MdpEncoder
from .base_decorator import EncoderDecorator


class MultiDecorator(EncoderDecorator):
    """A decorator class which allows to stack multiple decorators

    Parameters
    ----------
    encoder : MdpEncoder
        Encoder object to be wrapped
    decorators_and_kwargs : List[Tuple[Type[EncoderDecorator], Dict[str, Any]]]
        List of (decorator class - decorator argument) pairs.
        The first item will be the 'inner' decorator while the last item will
        be the 'outer' decorator.
    *args, **kwargs
        Any other argument which is forwarded to every decorator and 
        super().__init__()

    Attributes
    ----------
    unwrapped : MdpEncoder
        The wrapped encoder object
    """
    def __init__(self, encoder: MdpEncoder, decorators_and_kwargs: List[Tuple[Type[EncoderDecorator], Dict[str, Any]]], *args, **kwargs) -> None:
        for decorator_type, decorator_kwargs in decorators_and_kwargs:
            encoder = decorator_type(encoder, *args, **decorator_kwargs, **kwargs)
        super().__init__(encoder, *args, **kwargs)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, terminals: torch.Tensor) -> torch.distributions.Distribution:
        return self._wrapped_encoder.forward(observations, actions, rewards, next_observations, terminals)
