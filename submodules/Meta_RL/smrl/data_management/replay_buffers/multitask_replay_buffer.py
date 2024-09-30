"""
This module contains the class ``MultiTaskReplayBuffer``.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-26
"""

import numpy as np
import gym
from typing import List, Any, Dict

from .context_replay_buffer import ContextReplayBuffer
from smrl.utility.console_strings import warning


class MultiTaskReplayBuffer(ContextReplayBuffer):
    """A replay buffer which can also provide context data and encodings
    for each sample in the batch.

    The buffer stores the data twice (which is slightly memory-inefficient):
    As single samples and based on task ids. 

    This buffer is useful if we can assume that samples from one trajectory share
    the same task.

    ``random_batch()`` selects samples and their contexts (recent timesteps)
    from all observations (as if there was only one trajectory).

    ``random_context_target_batch()`` selects random contexts and targets from 
    the *same tasks*.
    You can determine if they are sampled in correct time-ordering
    or if they are sampled randomly from a trajectory be passing ``randomize_contexts``
    and/or ``randomize_targets``. If both values are ``False``, they are sampled
    in time-ordering and the targets are the next timesteps of the contexts.

    Parameters
    ----------
    max_replay_buffer_size : int
        Maximum number of transitions that the buffer can store.
    max_sub_size : int
        Maximum size of a subordinary task buffer
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
        max_replay_buffer_size: int, 
        max_sub_size: int,
        env: gym.Env, 
        encoding_dim: int, 
        randomize_contexts: bool = False,
        randomize_targets: bool = True,
        replace: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            encoding_dim=encoding_dim,
            randomize_contexts=randomize_contexts,
            randomize_targets=randomize_targets,
            replace=replace,
            *args,
            **kwargs,
        )

        self._task_buffers: List[ContextReplayBuffer] = []
        self._task_ids: List[Any] = []  # keeps track of task ids in self._task_buffers
        self._max_sub_size = max_sub_size

        print(warning(f"The selected replay buffer uses the privileged information ``task``."))

    def __repr__(self) -> str:
        repr_str = f"MultiTaskReplayBuffer with " \
            + f"{min(self._collected_samples, self._max_replay_buffer_size)}/{self._max_replay_buffer_size} " \
            + f"samples and {len(self._task_buffers)} task buffers."
        return repr_str

    def __str__(self):
        return self.__repr__()

    def add_sample(self, observation, action, reward, terminal, next_observation, encoding, task: Dict, **kwargs):
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
        task : Dict
            Task description
        """
        super().add_sample(observation, action, reward, terminal, next_observation, encoding, task, **kwargs)
        id = task['id']
        if id in self._task_ids:
            task_buffer = self._task_buffers[self._task_ids.index(id)]
        else:
            task_buffer = ContextReplayBuffer(
                max_replay_buffer_size=self._max_sub_size, 
                env=self.env, 
                encoding_dim=self._encoding_dim, 
                randomize_contexts=self.randomize_contexts, 
                randomize_targets=self.randomize_targets, 
                replace=self._replace, 
            )
            self._task_buffers.append(task_buffer)
            self._task_ids.append(id)
        task_buffer.add_sample(observation, action, reward, terminal, next_observation, encoding, task, **kwargs)
    
    def random_context_target_batch(self, batch_size: int, context_size: int, target_size: int) -> Dict[str, Dict[str, np.ndarray]]:
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
        # Collect batches from task buffers
        tasks: List[ContextReplayBuffer] = np.random.choice(
            self._task_buffers, size=batch_size,
            replace=(self._replace or batch_size > len(self._task_buffers))
        )
        batches = [t.random_context_target_batch(
                1, context_size, target_size, 
                self.randomize_contexts, self.randomize_targets
            ) for t in tasks]

        # Accumulate batch data
        contexts, targets = {}, {}
        for key in batches[0]['context'].keys():
            contexts[key] = np.concatenate([batch['context'][key] for batch in batches], axis=0)
            targets[key] = np.concatenate([batch['target'][key] for batch in batches], axis=0)

        return {'context': contexts, 'target': targets}

    def get_diagnostics(self):
        stats = {
            'number task replay buffers': len(self._task_buffers),
            'number of known task ids': len(self._task_ids),
            'total number of collected samples': self._collected_samples,
        }
        return stats
