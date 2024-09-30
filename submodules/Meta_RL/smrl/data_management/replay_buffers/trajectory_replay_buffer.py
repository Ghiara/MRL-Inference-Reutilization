"""
This module contains the class `TrajectoryReplayBuffer`.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-26
"""

import numpy as np
from typing import Dict, List
import gym
from collections import deque

from .context_replay_buffer import ContextReplayBuffer


class TrajectoryReplayBuffer(ContextReplayBuffer):
    """A replay buffer which can also provide context data and encodings
    for each sample in the batch.

    This buffer is useful if we can assume that samples from one trajectory share
    the same task.

    ``random_batch()`` selects samples and their contexts (recent timesteps)
    from all observations (as if there was only one trajectory).

    ``random_context_target_batch()`` selects contexts and targets from 
    trajectories. You can determine if they are sampled in correct time-ordering
    or if they are sampled randomly from a trajectory be passing ``randomize_contexts``
    and/or ``randomize_targets``. If both values are ``False``, they are sampled
    in time-ordering and the targets are the next timesteps of the contexts.

    Parameters
    ----------
    max_path_number : int
        Maximum number of paths that the buffer can store.
    max_sub_size : int
        Maximum size of a subordinary path buffer, should be roughly equal to the maximum path length
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
        by default True
    replace : bool, optional
        Determines if samples are drawn with replacement (True) or not (False)
    """

    def __init__(
        self, 
        max_path_number: int, 
        max_sub_size: int,
        env: gym.Env, 
        encoding_dim: int, 
        randomize_contexts: bool = False,
        randomize_targets: bool = True,
        replace: bool = True,
        max_replay_buffer_size: int = 0,    # should not be provided but is catched here 
                                            # to avoid double for super().__init__()
        *args,
        **kwargs,
    ):
        super().__init__(
            max_replay_buffer_size=0,
            env=env,
            encoding_dim=encoding_dim,
            randomize_contexts=randomize_contexts,
            randomize_targets=randomize_targets,
            replace=replace,
            *args,
            **kwargs,
        )

        self._path_buffers: List[ContextReplayBuffer] = deque([], maxlen=max_path_number)
        self._max_sub_size = max_sub_size
        self._collected_paths = 0   # Total number of collected paths

    def __repr__(self) -> str:
        repr_str = f"TrajectoryReplayBuffer with " \
            + f"{len(self._path_buffers)} path buffers."
        return repr_str

    def __str__(self):
        return self.__repr__()

    def random_batch(self, batch_size: int, context_size: int) -> Dict:
        """
        Returns a batch of training data which also includes the histories 
        (prior transitions) for each sample.

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
        paths: List[ContextReplayBuffer] = np.random.choice(
            self._path_buffers, size=batch_size,
            replace=(self._replace or batch_size > len(self._path_buffers))
        )
        batches = [p.random_batch(1, context_size) for p in paths]

        # Accumulate batch data
        batch = {'context': {}}
        for key in ['observations', 'actions', 'rewards', 'next_observations', 'terminals']:
            batch[key] = np.concatenate([b[key] for b in batches], axis=0)
            batch['context'][key] = np.concatenate([b['context'][key] for b in batches], axis=0)

        return batch

    def random_context_target_batch(
        self, 
        batch_size: int, 
        context_size: int, 
        target_size: int, 
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns a batch of training data (context & prediction target) for an
        inference mechanism.

        The context can be used for encoding the current MDP while the target
        is used for computing the prediction error.

        You can determine if the samples are selected in time-ordering
        or if they are sampled randomly. Use the properties ``self.randomize_contexts``
        and/or ``self.randomize_targets``. 
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

        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            'context': Context dictionary, arrays have shape (batch_size, context_size, *)
            'target': Target dictionary , arrays have shape (batch_size, target_size, *)
            Both dictionaries have keys: 'observations', 'actions', 'rewards', 'next_observations', 'terminals'
        """
        # Collect batches from path buffers
        paths: List[ContextReplayBuffer] = np.random.choice(
            self._path_buffers, size=batch_size,
            replace=(self._replace or batch_size > len(self._path_buffers))
        )
        batches = [p.random_context_target_batch(
                1, context_size, target_size, 
                self.randomize_contexts, self.randomize_targets
            ) for p in paths]

        # Accumulate batch data
        contexts, targets = {}, {}
        for key in batches[0]['context'].keys():
            contexts[key] = np.concatenate([batch['context'][key] for batch in batches], axis=0)
            targets[key] = np.concatenate([batch['target'][key] for batch in batches], axis=0)

        return {'context': contexts, 'target': targets}

    def add_path(self, path):
        self._collected_paths += 1
        self._collected_samples += len(path['observations'])
        new_path_buffer = ContextReplayBuffer(
            self._max_sub_size, 
            self.env, 
            self._encoding_dim, 
            randomize_contexts=self.randomize_contexts,
            randomize_targets=self.randomize_targets,
            replace=self._replace,
        )
        new_path_buffer.add_path(path)
        self._path_buffers.append(new_path_buffer)

    def get_diagnostics(self):
        stats = {
            'number of path replay buffers': len(self._path_buffers),
            'total number of collected samples': self._collected_samples,
            'total number of collected paths': self._collected_paths,
        }
        return stats
