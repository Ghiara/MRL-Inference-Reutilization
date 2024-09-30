"""
This module contains utility functions for data management: 
- `extract_history_data()`

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2022-12-02
"""

import numpy as np
from typing import List, Tuple


def extract_history_data(
    observations: List[np.ndarray],
    actions: List[np.ndarray],
    rewards: List[float],
    next_observations: List[np.ndarray],
    terminals: List[bool],
    context_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate history tensors of a specific lenght
    from the lists of observations, actions, rewards, and next states.

    NOTE: The length of the tensors may vary if the history data is 
    shorter than `context_size`. (The output is not padded!)

    Parameters
    ----------
    observations : List[np.ndarray]
        List of all observations from the current trajectory.
    actions : List[np.ndarray]
        List of all actions from the current trajectory.
    rewards : List[float]
        List of all rewards from the current trajectory.
    next_observations : List[np.ndarray]
        List of all 'next' observations from the current trajectory.
    terminals : List[bool]
        List of terminal indicators (~ end of trajectory).
    context_size : int
        Length of the history.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Arrays for observations, actions, rewards, next observations, and terminal indicators
    """
    observations = observations[-context_size:]
    actions = actions[-context_size:]
    rewards = rewards[-context_size:]
    next_observations = next_observations[-context_size:]
    terminals = terminals[-context_size:]
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    next_observations = np.array(next_observations, dtype=np.float32)
    terminals = np.array(terminals, dtype=np.bool8)

    # Make sure that all arrays have shape (sequence_length, <dim>)
    if len(observations.shape) == 1:
        observations = observations.reshape(-1, max(1, observations.shape[-1]))
    if len(actions.shape) == 1:
        actions = actions.reshape(-1, max(1, actions.shape[-1]))
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    if len(next_observations.shape) == 1:
        next_observations = next_observations.reshape(-1, max(1, next_observations.shape[-1]))
    if len(terminals.shape) == 1:
        terminals = terminals.reshape(-1, 1)

    return observations, actions, rewards, next_observations, terminals


def pad_context_data(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_observations: np.ndarray,
    terminals: np.ndarray,
    context_size: int,
    padding_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pad context data of length l < context_size to have exactly length = context_size.

    Parameters
    ----------
    observations : np.ndarray
        Observation context data, Shape: (l, observation_dim)
    actions : np.ndarray
        Action context data, Shape: (l, action_dim)
    rewards : np.ndarray
        Reward context data, Shape: (l, 1)
    next_observations : np.ndarray
        Next observations context data, Shape: (l, observation_dim)
    terminals : np.ndarray
        Terminal indicator context data, Shape: (l, 1)
    context_size : int
        Length of the history. Context information vectors will be padded to
        have the length context_size.
    padding_value : float, optional
        The value which is used for padding, by default zero.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Arrays for observations, actions, rewards, next observations, and terminal indicators
        with exactly the length context_size
    """

    l = observations.shape[0]
    assert (actions.shape[0] == l
            and rewards.shape[0] == l 
            and next_observations.shape[0] == l
            and terminals.shape[0] == l), "Context data is corrupted! (Inputs do not have the same length.)"

    pad = context_size - l
    if pad > 0:
        observations = np.pad(observations, ((pad, 0), (0, 0)), constant_values=padding_value)
        actions = np.pad(actions, ((pad, 0), (0, 0)), constant_values=padding_value)
        rewards = np.pad(rewards, ((pad, 0), (0, 0)), constant_values=padding_value)
        next_observations = np.pad(next_observations, ((pad, 0), (0, 0)), constant_values=padding_value)
        terminals = np.pad(terminals, ((pad, 0), (0, 0)), constant_values=False)

    return observations, actions, rewards, next_observations, terminals