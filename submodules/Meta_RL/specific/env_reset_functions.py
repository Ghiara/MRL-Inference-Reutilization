"""
This file implements environment resets functions which can overwrite the 
default reset function of an environment when using ``ContextCollector`` (see
smrl>data_management>vae_training_data.py). They have been replaced by an
improved reset scheme of the environments and are not strictly required any more.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-03-13

Note:
    <Comments which might be important, e.g. 
    where an original version of the file was found.>
"""

import numpy as np


def toy1d_borderless_reset() -> np.ndarray:
    """
    Reset function for borderless Toy1D
    """
    return np.random.randn(1) * 50


def toy1d_borderless_small_reset() -> np.ndarray:
    """
    Reset function for borderless Toy1D
    """
    return np.random.randn(1) * 1

