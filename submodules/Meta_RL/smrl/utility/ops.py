"""
This module contains utility functions which can be reused in multiple places.
It contains the following functions:
- circular_slicing()
- np_batch_to_tensor_batch()

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-24
"""

import numpy as np
import torch
import importlib
from typing import List, Union, Dict, Any, Type, Callable
from enum import Enum
import json
import collections.abc

import rlkit.torch.pytorch_util as ptu

def circular_slicing(lst: Union[List, np.ndarray, torch.Tensor], index_low: int, index_high: int, min_index: int = None, max_index: int = None) -> Union[List, np.ndarray, torch.Tensor]:
    """Allows circular indexing, e.g. 
    ```
    >>> lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> index_low = -2
    >>> index_high = 3
    >>> circular_slicing(lst, index_low, index_high)
    [8, 9, 0, 1, 2]
    ```

    With a circular slice, indices can exceed the bounds of the list (index_high > len(lst))
    and a combination of index_low < 0, index_high >= 0 is possible.
    The list is interpreted as if it would repeat infinitely.

    Parameters
    ----------
    lst : Union[List, np.ndarray, torch.Tensor]
        List (or array / tensor) which should be indexed
    index_low : int
        High index
    index_high : int
        Low index
    min_index : int, optional
        If specified, only the part of the list which is higher than min_index is considered.
    max_index : int, optional
        If specified, only the part of the list which is smaller than max_index is considered. 

    Returns
    -------
    Union[List, np.ndarray, torch.Tensor]
        The sliced part of the list, same type as input lst
    """

    lst_type = type(lst)

    if max_index is None:
        max_index = len(lst)
    if min_index is None:
        min_index = 0
    assert 0 <= max_index <= len(lst), "max_index cannot exceed the length of the list and cannot be smaller than 0!"
    assert 0 <= min_index <= len(lst), "min_index cannot be smaller than 0 and cannot exceed the length of the list!"
    assert min_index <= max_index, "min_index cannot be larger than max_index!"

    if index_low > index_high:
        return lst_type([])

    # Make sure that 0 <= index_low < len(lst)
    # without losing distance betwen index_low and index_high
    shift = (index_low % len(lst)) - index_low
    index_low += shift
    index_high += shift

    # Generate indices fro index_low -> index_high such that they are
    # within the list range
    index_low -= min_index
    index_high -= min_index
    indices = [(i % max_index) + min_index for i in range(index_low, index_high)]

    if isinstance(lst, np.ndarray):
        lst_indexed = lst[indices]
    elif isinstance(lst, torch.Tensor):
        lst_indexed = lst[indices]
    else:
        lst_indexed = [lst[i] for i in indices]
    return lst_indexed


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
            np_batch[key] = ptu.from_numpy(value)
        if isinstance(value, dict):
            np_batch[key] = np_batch_to_tensor_batch(value)
    return np_batch


def unnest_dictionary(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Unnest a nested dictionary such that it has only one level.

    All sublevel keys are mapped to a combined key in which the hiearchy is 
    marked with slashes "/".

    Example:
    ```
    >>> nested_dict = {'a': 1, 'b': {'b.1': 'i', 'b.2': 'ii'}}
    >>> unnested_dict = unnest_dictionary(nested_dict)
    >>> # Output: 
    >>> # {'a': 1, 'b/b.1': 'i', 'b/b.2': 'ii'}
    ```

    Parameters
    ----------
    dictionary : Dict[str, Any]
        A (potentially nested) dictionary

    Returns
    -------
    Dict[str, Any]
        Unnested dictionary
    """
    unnested_dictionary = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            sub_dict = unnest_dictionary(value)
            for k, v in sub_dict.items():
                unnested_dictionary[key + "/" + k] = v
        else:
            unnested_dictionary[key] = value
    return unnested_dictionary


def deep_dictionary_update(dictionary: Dict, update_dictionary: Dict):
    """
    Deep dictionary update. 
    
    Instead of replacing values entirely (as in dict.update), 
    nested dictionary values are updated without overwriting them entirely.

    NOTE: The original dictionary object ``dictionary`` is modified!
    To avoid this, use ``copy.deepcopy(dictionary)``.

    References
    ----------
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    Parameters
    ----------
    dictonary : Dict
        Dictionary which is to be updated
    update_dictionary : Dict
        Dictionary which is used for updating
    """
    if not isinstance(dictionary, collections.abc.Mapping):  
        # happens if update_dictionary has a nested dictionary where dictionary 
        # is not nested
        return update_dictionary
    
    for k, v in update_dictionary.items():
        if isinstance(v, collections.abc.Mapping):
            # recursive call
            dictionary[k] = deep_dictionary_update(dictionary.get(k, {}), v)
        else:
            # assignment
            dictionary[k] = v
    return dictionary


def type_from_json_dict(dictionary: Dict[str, str]) -> Union[Type, Callable]:
    """Get a class or function from a json-dumped class/function attribute.

    Class attributes in dictionaries are mapped to dictionary entries
    ```
    {'$class': '<module_name>.<class_name>'}
    ```
    which cannot be restored to classes directly.
    Similarly, functions in dictionaries are mapped to dictionary entries
    ```
    {'$function': '<module_name>.<function_name>'}
    ```
    which cannot be restored to functions directly.

    This function restores classes and functions from strings by importing the module.

    Parameters
    ----------
    dictionary : Dict[str, str]
        A dictionary with structure ``{'$class': '<module_name>.<class_name>'}``
        OR ``{'$function': '<module_name>.<function_name>'}``

    Returns
    -------
    Union[Type, Callable]
        The class or function
    """
    for key in ('$class', '$function'):
        if not key in dictionary.keys():
            continue
        module_and_type = dictionary[key]
        type_name = module_and_type.split(".")[-1]
        module_name = module_and_type.removesuffix("." + type_name)
        module = importlib.import_module(module_name)
    return getattr(module, type_name)


def ensure_importable_entries(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Map all entries
    ```
    {'$class': '<module_name>.<class_name>'}
    ```
    or
    ```
    {'$function': '<module_name>.<function_name>'}
    ```
    in the dictionary to classes/functions, respectively.

    The above entry type is found if a dictionary with class or function entries 
    is dumped to a json-file and loaded afterwards.

    Parameters
    ----------
    dictionary : Dict[str, Any]
        Dictionary, potentially loaded from json, which may have "$class" and 
        "$function" keys.

    Returns
    -------
    Dict[str, Any]
        Dictionary where all "$class" and "$function" keys were mapped to classes/functions.
    """
    if isinstance(dictionary, dict):
        if "$class" in dictionary.keys():
            dictionary = type_from_json_dict(dictionary)
        elif "$function" in dictionary.keys():
            dictionary = type_from_json_dict(dictionary)
        else: 
            for key, value in dictionary.items():
                dictionary[key] = ensure_importable_entries(value)
        return dictionary
    elif isinstance(dictionary, list):
        return [ensure_importable_entries(item) for item in dictionary]
    elif isinstance(dictionary, tuple):
        return tuple([ensure_importable_entries(item) for item in dictionary])
    else:
        return dictionary


class CustomJsonEncoder(json.JSONEncoder):
    """Allows customized json-dumping for classes, functions, and enums.

    Reference
    ---------
    RLKIT: rlkit > core > logging.py

    Usage
    -----
    ```
    json.dump(<dictionary>, ..., cls=MyEncoder)
    ```
    """
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)
