"""
This module defines samplers and buffers for encoder training data (context + target).
It can be used with the stand-alone inference training algorithm ``MdpVaeAlgorithm``
(see algorithms/encoder_algorithm.py).

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-03-16
"""

from typing import Tuple, Dict, List, Any, Callable
import numpy as np
from collections import deque

from smrl.environments.state_access_env import StateAccessMetaEnv
from smrl.policies.base import Policy
from smrl.utility.console_strings import warning


class ContextTargetBuffer():
    """A buffer for MdpVAE training which provides context-target pairs.

    Parameters
    ----------
    maxlen : int
        Maximum number of samples that the buffer can store.
    """
    def __init__(self, maxlen: int) -> None:
        self.contexts: List[Dict[str, np.ndarray]] = deque([], maxlen=maxlen)
        self.targets: List[Dict[str, np.ndarray]] = deque([], maxlen=maxlen)
        self.maxlen = maxlen
        self.total_samples = 0

    def add_samples(self, contexts: List[Dict[str, np.ndarray]], targets: List[Dict[str, np.ndarray]], *args, **kwargs):
        """Add samples to the buffer

        Parameters
        ----------
        contexts : List[Dict[str, np.ndarray]]
            List of new contexts which should be added to the buffer.
        targets : List[Dict[str, np.ndarray]]
            List of new targets (belonging to the contexts) which should be
            added to the buffer.
        """
        assert len(contexts) == len(targets), "Inputs must have the same number of elements!"
        self.contexts.extend(contexts)
        self.targets.extend(targets)
        self.total_samples += len(contexts)

    def random_batch(self, batch_size: int, *args, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Sample a random batch of context and target data from the buffer.

        Parameters
        ----------
        batch_size : int
            Batch size
        
        Returns
        -------
        contexts : Dict[str, np.ndarray]
            Dictionary with keys 'observations', 'actions', 'rewards', 
            'next_observations', 'terminals'. 
            Each entry has shape (batch_size, context_size, *)
        targets : Dict[str, np.ndarray]
            Dictionary with keys 'observations', 'actions', 'rewards', 
            'next_observations', 'terminals'. 
            Each entry has shape (batch_size, target_size, *)
        """
        idcs = np.random.choice(len(self.contexts), size=batch_size, replace=True)
        contexts, targets = {}, {}
        for key in ['observations', 'actions', 'rewards', 'next_observations', 'terminals']:
            contexts[key] = np.concatenate([self.contexts[i][key][None] for i in idcs], axis=0)
            targets[key] = np.concatenate([self.targets[i][key][None] for i in idcs], axis=0)
        return contexts, targets

    def get_diagnostics(self) -> Dict[str, Any]:
        stats = dict(
            buffered_samples = len(self.contexts),
            total_samples = self.total_samples,
        )
        return stats
        

class ContextTargetTaskBuffer(ContextTargetBuffer):
    """A buffer for MdpVAE training which provides context-target pairs.

    Context and target can be different samples which belong to the same task.

    Parameters
    ----------
    maxlen : int
        Maximum number of samples that the buffer can store.
    """
    def __init__(self, maxlen: int) -> None:
        super().__init__(maxlen)
        self.contexts: Dict[int, List[Dict[str, np.ndarray]]] = {}  # Dictionary with list of contexts for each task
        self.targets: Dict[int, List[Dict[str, np.ndarray]]] = {}   # Dictionary with list of targets for each task

    def add_samples(self, contexts: List[Dict[str, np.ndarray]], targets: List[Dict[str, np.ndarray]], tasks: List[int], *args, **kwargs):
        """Add samples to the buffer

        Parameters
        ----------
        contexts : List[Dict[str, np.ndarray]]
            List of new contexts which should be added to the buffer.
        targets : List[Dict[str, np.ndarray]]
            List of new targets (belonging to the contexts) which should be
            added to the buffer.
        tasks : List[int]
            List of task ids for each context-target pair
        """
        assert len(contexts) == len(targets) == len(tasks), "Inputs must have the same number of elements!"
        for context, target, task in zip(contexts, targets, tasks):
            try:
                self.contexts[task].append(context)
                self.targets[task].append(target)
            except KeyError:
                self.contexts[task] = deque([context], maxlen=self.maxlen)
                self.targets[task] = deque([target], maxlen=self.maxlen)
        self.total_samples += len(contexts)

    def random_batch(self, batch_size: int, *args, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        tasks = np.random.choice(list(self.contexts.keys()), size=batch_size, replace=True)   # Sample uniformly from all known tasks
        idcs_c = [np.random.choice(len(self.contexts[t])) for t in tasks]   # For each task, generate index for the corresponding contexts
        idcs_t = [np.random.choice(len(self.targets[t])) for t in tasks]    # For each task, generate index for the corresponding targets, INDPENDENT from the context indices
        contexts, targets = {}, {}
        for key in ['observations', 'actions', 'rewards', 'next_observations', 'terminals']:
            contexts[key] = np.concatenate([self.contexts[t][i][key][None] for t, i in zip(tasks, idcs_c)], axis=0)
            targets[key] = np.concatenate([self.targets[t][i][key][None] for t, i in zip(tasks, idcs_t)], axis=0)
        return contexts, targets

    def get_diagnostics(self) -> Dict[str, Any]:
        stats = dict(
            buffered_samples = sum([len(task_contexts) for task_contexts in self.contexts.values()]),
            total_samples = self.total_samples,
            tasks = len(self.contexts.keys()),
        )
        return stats


class ContextCollector():
    """A data collector which samples context and target data for MdpVAE training.

    The samples are generated by 
    1) Randomly resetting the state of the environment by sampling from the
        observation space.
    2) Resetting the environment task
    3) Rolling out a *short* trajectory of length context_size + target_size

    Parameters
    ----------
    env : StateAccessMetaEnv
        Environment which allows to modify the state from outside
    context_size : int
        Size of the context samples
    target_size : int
        Size of the target samples
    policy : Policy, optional
        Policy for rollouts, by default None
    reset_function : Callable[[], np.ndarray]
        Function for sampling start states before each context-target rollout.
        By default, the observation-space's ``sample()`` function (of the 
        environment) is used.
    incomplete_context_prob : float, optional
        Probability of sampling an incomplete context which has a length < context_size.
        Missing timesteps are filled with zeros. The length of the incomplete
        context is sampled uniformly from range(context_size).
        By default 0.0
    pad_kwargs : Dict[str, Any]
        Arguments for ``numpy.pad`` when padding incomplete trajectories.
        Only relevant if incomplete_context_prob > 0
    """
    def __init__(
            self, 
            env: StateAccessMetaEnv, 
            context_size: int, 
            target_size: int, 
            policy: Policy = None, 
            reset_function: Callable[[], np.ndarray] = None,
            incomplete_context_prob: float = 0.0,
            pad_kwargs: Dict[str, Any] = None,
        ) -> None:
        self.env = env
        self.policy = policy
        self.context_size = context_size
        self.target_size = target_size
        self.incomplete_context_prob = incomplete_context_prob
        self.pad_kwargs = pad_kwargs if pad_kwargs is not None else {}

        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape
        self.rew_shape = (1, )

        if reset_function is None:
            # reset_function = env.observation_space.sample
            reset_function = lambda: env.reset()[0]
        self.reset_function = reset_function


    def collect_context_and_target(self, policy: Policy = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
        """Collect one context-target pair

        Parameters
        ----------
        policy : Policy, optional
            Policy for rollout. If not provided self.policy will be taken.
        
        Returns
        -------
        context : Dict[str, np.ndarray]
            Dictionary with keys 'observations', 'actions', 'rewards', 
            'next_observations', 'terminals'. 
            Each entry has shape (context_size, *)
        target : Dict[str, np.ndarray]
            Dictionary with keys 'observations', 'actions', 'rewards', 
            'next_observations', 'terminals'. 
            Each entry has shape (target_size, *)
        task_id : int
            Id of the true task which was used for rollout
        """
        if policy is None: policy = self.policy

        if np.random.rand() < self.incomplete_context_prob:
            context_size = np.random.choice(self.context_size)
        else:
            context_size = self.context_size

        context = dict(
            observations = np.zeros([context_size, *self.obs_shape]),
            actions = np.zeros([context_size, *self.act_shape]),
            next_observations = np.zeros([context_size, *self.obs_shape]),
            rewards = np.zeros([context_size, *self.rew_shape]),
            terminals = np.zeros([context_size, 1]),
        )
        targets = dict(
            observations = np.zeros([self.target_size, *self.obs_shape]),
            actions = np.zeros([self.target_size, *self.act_shape]),
            next_observations = np.zeros([self.target_size, *self.obs_shape]),
            rewards = np.zeros([self.target_size, *self.rew_shape]),
            terminals = np.zeros([self.target_size, 1]),
        )

        policy.reset()
        self.env.state = self.reset_function()
        self.env.sample_task()

        obs = self.env.observation
        task_id = self.env.task['id']
        for i in range(context_size):
            act, _ = policy.get_action(obs)
            n_obs, rew, term, trunc, info = self.env.step(act)
            context['observations'][i] = obs
            context['actions'][i] = act
            context['next_observations'][i] = n_obs
            context['rewards'][i] = rew
            context['terminals'][i] = term
            obs = n_obs
            if self.env.task['id'] != task_id:
                print(warning("The task has changed during sampling! Ideally, the task should not change within context_size + target_size steps."))
        for i in range(self.target_size):
            act, _ = policy.get_action(obs)
            n_obs, rew, term, trunc, info = self.env.step(act)
            targets['observations'][i] = obs
            targets['actions'][i] = act
            targets['next_observations'][i] = n_obs
            targets['rewards'][i] = rew
            targets['terminals'][i] = term
            obs = n_obs
            if self.env.task['id'] != task_id:
                print(warning("The task has changed during sampling! Ideally, the task should not change within context_size + target_size steps."))

        if context_size == 0:
            context = dict(
                observations = np.zeros([self.context_size, *self.obs_shape]),
                actions = np.zeros([self.context_size, *self.act_shape]),
                next_observations = np.zeros([self.context_size, *self.obs_shape]),
                rewards = np.zeros([self.context_size, *self.rew_shape]),
                terminals = np.zeros([self.context_size, 1]),
            )
        elif context_size < self.context_size:
            for key, value in context.items():
                context[key] = np.pad(value, ((self.context_size - context_size, 0), *[(0, 0) for _ in range(value.ndim - 1)]), **self.pad_kwargs)

        return context, targets, task_id

    
    def collect_data(self, n_samples: int, policy: Policy = None) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], List[int]]:
        """Collect multiple context-target pairs
        
        Parameters
        ----------
        n_samples : int
            Number of context-target pairs
        policy : Policy, optional
            Policy for rollout. If not provided self.policy will be taken.
        
        Returns
        -------
        contexts : List[Dict[str, np.ndarray]]
            List of context dictionaries with keys 'observations', 'actions', 
            'rewards', 'next_observations', 'terminals'. 
            Each entry has shape (context_size, *)
        targets : List[Dict[str, np.ndarray]]
            List of target dictionaries with keys 'observations', 'actions', 
            'rewards', 'next_observations', 'terminals'. 
            Each entry has shape (target_size, *)
        tasks : int
            Ids of the true tasks which was used for rollout
        """
        contexts, targets, tasks = [], [], []
        for _ in range(n_samples):
            c, t, task_id = self.collect_context_and_target(policy)
            contexts.append(c)
            targets.append(t)
            tasks.append(task_id)

        return contexts, targets, tasks
