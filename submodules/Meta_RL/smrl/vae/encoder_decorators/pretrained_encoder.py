"""
This module contains the class ``PretrainedEncoder`` and the factory function
``pretrained_encoder()``.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-26
"""

import torch

from ..mdpvae import MdpEncoder
from .base_decorator import EncoderDecorator


class PretrainedEncoder(EncoderDecorator):
    """An encoder decorator which wraps a pretrained encoder.

    Parameters
    ----------
    encoder : MdpEncoder
        The wrapped encoder 
    path_to_weights : str
        Path to the pretrained encoder's weights 
    state_dict_keyword : str, optional
        Keyword for the state_dict from which the pretrained encoder weights
        are loaded. Set to None to indicate that the state_dict contains only
        the encoder weights.
        By default 'trainer/Inference trainer/encoder'
    trainable : bool, optional
        Set to ``True`` to train the wrapped encoder further,
        by default False 
    *args, **kwargs
        Instantiation arguments for the wrapped encoder object
    """

    def __init__(self, encoder: MdpEncoder, path_to_weights: str, state_dict_keyword: str = 'trainer/Inference trainer/encoder', trainable: bool = False, *args, **kwargs) -> None:
        super().__init__(encoder, *args, **kwargs)
        state_dict = torch.load(path_to_weights, map_location=torch.device('cpu'))
        if state_dict_keyword is not None:
            self.unwrapped.load_state_dict(state_dict[state_dict_keyword])
        else:
            self.unwrapped.load_state_dict(state_dict)
        print("Loaded state_dict of pretrained encoder.")
        self.trainable = trainable

    def load_state_dict(self, *args, **kwargs):
        if not self.trainable:
            print("PretrainedEncoder has not loaded the new state_dict. " \
                + "This is intended since state_dict has been loaded at instantiation.")
        else:
            super().load_state_dict(*args, **kwargs)
            print("Loaded new state dict (which might differ from pretrained state dict)"
                + " because encoder is trainable.")

    def forward(self, *args, **kwargs):
        if self.trainable:
            return self._wrapped_encoder.forward(*args, **kwargs)
        else:
            with torch.no_grad():
                return self._wrapped_encoder.forward(*args, **kwargs)

