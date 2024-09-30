"""
This module contains utility functions for pytorch.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2022-12-12

Note: The functions are modified versions of the ones from RLKIT, see
    https://github.com/rail-berkeley/rlkit
"""

import torch
import numpy as np

def torch_ify(x, *args, **kwargs) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, *args, **kwargs).float()

def np_ify(x, *args, **kwargs) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    else:
        return np.array(x)

def np_batch_to_tensor_batch(np_batch: dict) -> dict:
    """Accepts a batch dictionary with entries with np.ndarray entries
    and dictionary entries (nesting).

    Every numpy array is mapped to a torch Tensor.

    Parameters
    ----------
    np_batch : dict
        Dictioray with np.ndarray or nested dictionaries.

    Returns
    -------
    dict
        Dictionary with same structure as input array where all numpy arrays
        have been replaced with torch Tensors.
    """
    for key, value in np_batch.items():
        if isinstance(value, np.ndarray):
            np_batch[key] = torch_ify(value)
        if isinstance(value, dict):
            np_batch[key] = np_batch_to_tensor_batch(value)
    return np_batch