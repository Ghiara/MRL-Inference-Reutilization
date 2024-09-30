"""
This module contains the class ``EncoderDecorator`` which is a base class
for encoder decorators, and the factory function ``encoder_decorator()``.

Encoder decorators implement a variant of the decorator pattern, see 
https://en.wikipedia.org/wiki/Decorator_pattern or
https://python-patterns.guide/gang-of-four/decorator-pattern/.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-19
"""

import torch
from typing import Type, Any, Mapping

from smrl.utility.console_strings import warning

from ..mdpvae import MdpEncoder


class EncoderDecorator(MdpEncoder):
    """A decorator for encoders.

    Parameters
    ----------
    encoder : MdpEncoder
        Encoder object to be wrapped
    *args, **kwargs
        Any other argument which is forwarded to MdpEncoder (the parent class
        of EncoderDecorator)

    Attributes
    ----------
    unwrapped : MdpEncoder
        The wrapped encoder object
    """
    # _wrapped_encoder: MdpEncoder = None

    def __init__(self, encoder: MdpEncoder, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wrapped_encoder = encoder
        self.supports_variable_sequence_length = encoder.supports_variable_sequence_length
        self._consistency_check()

    def _consistency_check(self):
        if self.encoding_mode != self._wrapped_encoder.encoding_mode:
            print(warning("Encoder decorator and decorated encoder do not have the same " +
                          f"encoding mode! ('{self.encoding_mode}' and '{self._wrapped_encoder.encoding_mode}')"))
        if self.encoding_dim != self._wrapped_encoder.encoding_dim:
            print(warning("Encoder decorator and decorated encoder do not have the same " +
                          f"encoding dimension! ({self.encoding_dim} and {self._wrapped_encoder.encoding_dim}"))
        if self.latent_dim != self._wrapped_encoder.latent_dim:
            print(warning("Encoder decorator and decorated encoder do not have the same " +
                          f"latent dimension! ({self.latent_dim} and {self._wrapped_encoder.latent_dim}"))
                        
    @property
    def unwrapped(self):
        """The undecorated encoder object.
        """
        if isinstance(self._wrapped_encoder, EncoderDecorator):
            return self._wrapped_encoder.unwrapped
        else:
            return self._wrapped_encoder

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self._wrapped_encoder.load_state_dict(state_dict, strict)

    def state_dict(self):
        return self._wrapped_encoder.state_dict()

    def to(self, device: torch.device):
        super().to(device)
        self._wrapped_encoder.to(device)

    def train(self, mode: bool = True):
        self._wrapped_encoder.train(mode)
        return super().train(mode)

    def __getattr__(self, name: str):
        # By default, any non-provided value and function is first passed on to 
        # super(), then to the wrapped object.
        # This is reasonable because it ensures that any derived subclasses 
        # (multilevel inheritance) first ask their parent classes for another
        # implementation.
        # All functions which should be handled by _wrapped_encoder should be stated
        # explicitely (see, e.g., 'load_state_dict()', 'train()', etc. above). 
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "_wrapped_encoder" in self.__dict__:
                return getattr(self._wrapped_encoder, name)
            else:
                raise AttributeError(f"{self.__class__} has no attribute {name}")

