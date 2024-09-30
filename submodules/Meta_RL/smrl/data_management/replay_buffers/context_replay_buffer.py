"""
This module contains the class `ContextReplayBuffer`.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-27
"""

import numpy as np
import gym
from typing import Dict
from collections import deque

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer

from smrl.utility.console_strings import warning


class ContextReplayBuffer(EnvReplayBuffer):
    """A replay buffer which can also provide context data and encodings
    for each sample in the batch.

    This class can be considered as the base class for context-providing replay
    buffers.

    ``random_context_target_batch()`` selects contexts and targets from 
    the data. You can determine if they are sampled in correct time-ordering
    or if they are sampled randomly from a trajectory be passing ``randomize_contexts``
    and/or ``randomize_targets``. If both values are ``False``, they are sampled
    in time-ordering and the targets are the next timesteps of the contexts.

    Parameters
    ----------
    max_replay_buffer_size : int
        Maximum number of transitions that the buffer can store.
    env : Env
        Environment, used for determining observation space dimension and action space dimension.
        (Required for memory initialization.)
    encoding_dim : int
        Dimension of the encodings. (Required for memory initialization.)
    randomize_contexts : bool, optional
        If True, context samples from ``random_context_target_batch()`` are not time-ordered,
        by default False
    randomize_targets : bool, optional
        If True, target samples from ``random_context_target_batch()`` are not time-ordered,
        by default False
    replace : bool, optional
        Determines if samples are drawn with replacement (True) or not (False)
    """
    def __init__(
        self, 
        max_replay_buffer_size: int, 
        env: gym.Env, 
        encoding_dim: int, 
        randomize_contexts: bool = False,
        randomize_targets: bool = False,
        replace: bool = True, 
        *args, 
        **kwargs,
    ):
        super().__init__(max_replay_buffer_size, env, *args, **kwargs)

        self._replace = replace
        self._encoding_dim = encoding_dim
        self.randomize_contexts = randomize_contexts
        self.randomize_targets = randomize_targets

        self._encodings = np.zeros((max_replay_buffer_size, encoding_dim))
        self._tasks = deque([], maxlen=max_replay_buffer_size)
        self._collected_samples = 0

    def __len__(self):
        return self._size

    def __repr__(self):
        return f"ContextReplayBuffer with {self._size} samples."

    def random_batch(self, batch_size: int, context_size: int) -> Dict:
        """
        Returns a batch of training data which also includes the histories 
        (prior transitions) for each sample.

        NOTE: Contexts from ``random_batch()`` are never randomized because
        they should resemble rollout data.

        Parameters
        ----------
        batch_size : int
            Batch size
        context_size : int
            Length of context sequence.

        Returns
        -------
        Dict
            Dictionary of transition data, including
            - observations : np.ndarray, shape (batch_size, *)
            - actions : np.ndarray, shape (batch_size, *)
            - rewards : np.ndarray, shape (batch_size, *)
            - terminals : np.ndarray, shape (batch_size, *)
            - next_observations : np.ndarray, shape (batch_size, *)
            - encodings : np.ndarray, shape (batch_size, *)
            - context : Dict[torch.Tensor], arrays of shape (batch_size, context_size, *)
        """
        # TODO: Figure out what to do if self._size < context_size!
        indices = np.random.choice(np.arange(min(context_size, self._size), self._size), size=batch_size, replace=self._replace or self._size < batch_size)
        # OPTION: The indices below might be more realistic as they 
        #   also sample batches from the very beginning of an episode
        # indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        
        # Construct batch from data and indices
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            encodings=self._encodings[indices],
            context=dict(
                observations=np.zeros((batch_size, context_size, *self._observations.shape[1:])),
                actions=np.zeros((batch_size, context_size, *self._actions.shape[1:])),
                rewards=np.zeros((batch_size, context_size, *self._rewards.shape[1:])),
                next_observations=np.zeros((batch_size, context_size, *self._next_obs.shape[1:])),
                terminals=np.zeros((batch_size, context_size, *self._terminals.shape[1:])),
            ),            
        )

        # Fill context data
        for i, index in enumerate(indices):
            t = np.arange(max(0, index - context_size), index)
            batch['context']['observations'][i] = np.pad(self._observations[t], ((context_size - len(t), 0), (0, 0)))
            batch['context']['actions'][i] = np.pad(self._actions[t], ((context_size - len(t), 0), (0, 0)))
            batch['context']['rewards'][i] = np.pad(self._rewards[t], ((context_size - len(t), 0), (0, 0)))
            batch['context']['next_observations'][i] = np.pad(self._next_obs[t], ((context_size - len(t), 0), (0, 0)))
            batch['context']['terminals'][i] = np.pad(self._terminals[t], ((context_size - len(t), 0), (0, 0)))

        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def random_context_target_batch(
        self, 
        batch_size: int, 
        context_size: int, 
        target_size: int, 
        randomize_contexts: bool = None, 
        randomize_targets: bool = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns a batch of training data (context & prediction target) for an
        inference mechanism.

        The context can be used for encoding the current MDP while the target
        is used for computing the prediction error.

        You can determine if the samples are selected in time-ordering
        or if they are sampled randomly. Use the arguments ``randomize_contexts``
        and/or ``randomize_targets``. 
        If both values are ``False``, they are sampled in time-ordering and the
        targets are the next timesteps of the contexts.

        Parameters
        ----------
        batch_size : int
            Size of the batch
        context_size : int
            Length of the context sequence -> Encoder!
        target_size : int
            Length of the target sequence
        randomize_contexts : bool
            If True, context samples are not time-ordered
        randomize_targets : bool
            If True, target samples are not time-ordered

        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            'context': Context dictionary, arrays have shape (batch_size, context_size, *)
            'target': Target dictionary , arrays have shape (batch_size, target_size, *)
            Both dictionaries have keys: 'observations', 'actions', 'rewards', 'next_observations', 'terminals'
        """
        if randomize_contexts is None: randomize_contexts = self.randomize_contexts
        if randomize_targets is None: randomize_targets = self.randomize_targets

        # Get center-times (used if not randomized)
        min_time, max_time = 0, self._size - target_size * (not randomize_targets)
        if min_time > max_time: # Not enough samples
            print(warning("This replay buffer doesn't have enough samples for the requested target size"))
            max_time = min_time
        replace = self._replace or (min_time - max_time) < batch_size
        if replace and not self._replace:
            print(warning("Sampling with replacement is temporarily active because this replay buffer does not have enough samples."))
        t = np.random.choice(np.arange(min_time, max(min_time, max_time)+1), size=batch_size, replace=replace) # First timestep of targets, next timestep of context
        
        # Sample indices for context and target
        context_indices = t[..., None] - np.arange(context_size, 0, -1)
        target_indices = t[..., None] + np.arange(target_size)
        if randomize_contexts:
            context_indices = np.random.choice(self._size, size=(batch_size, context_size))
        if randomize_targets:
            target_indices = np.random.choice(self._size, size=(batch_size, target_size))
        
        # Construct batches from data and indices
        corrected_context_indices = np.where(context_indices >= 0, context_indices, 0)
        context = dict(
            observations = self._observations[corrected_context_indices],
            actions = self._actions[corrected_context_indices],
            rewards = self._rewards[corrected_context_indices],
            next_observations = self._next_obs[corrected_context_indices],
            terminals = self._terminals[corrected_context_indices],
        )
        for key in context.keys():
            context[key][context_indices < 0] = 0
        corrected_target_indices = np.where(target_indices<self._size, target_indices, self._size-1)
        target = dict(
            observations = self._observations[corrected_target_indices],
            actions = self._actions[corrected_target_indices],
            rewards = self._rewards[corrected_target_indices],
            next_observations = self._next_obs[corrected_target_indices],
            terminals = self._terminals[corrected_target_indices],
        )
        for key in target.keys():
            # This case should be avoided as much as possible:
            # Predicting some target from no context (context has zeros) is bad but
            # predicting no target from some context (target has zeros) is simply wrong
            target[key][target_indices >= self._size] = 0

        return {'context': context, 'target': target}
    
    def add_path(self, path: Dict[str, np.ndarray]):
        """Add one path (multiple samples) to the internal data storage arrays.

        Parameters
        ----------
        path : Dict[str, np.ndarray]
            Dictionary with keys
            - "observations"
            - "actions"
            - "rewards"
            - "next_observations"
            - "terminals"
            - "encodings"
            - "agent_infos"
            - "env_infos"
            - "tasks"
        """
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                encoding,
                agent_info,
                env_info,
                task
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["encodings"],
            path["agent_infos"],
            path["env_infos"],
            path["tasks"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                encoding=encoding,
                agent_info=agent_info,
                env_info=env_info,
                task=task
            )

    def add_sample(self, observation, action, reward, terminal, next_observation, encoding, task, **kwargs):
        """Add one transition to the internal data storage arrays.

        Parameters
        ----------
        observation : np.ndarray
            Observation
        action : np.ndarray
            Action
        reward : np.ndarray
            Reward
        terminal : np.ndarray
            Termination indicator
        next_observation : np.ndarray
            Next observation
        encoding : np.ndarray
            Latent encoding
        """
        self._encodings[self._top] = encoding
        self._tasks.append(task)
        self._collected_samples += 1
        return super().add_sample(observation, action, reward, terminal, next_observation, **kwargs)
