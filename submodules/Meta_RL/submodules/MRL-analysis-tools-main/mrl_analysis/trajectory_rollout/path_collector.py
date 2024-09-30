"""
This module contains path collector classes which can sample multiple rollouts
and return the obtained paths. There is also a multithreaded variant based on ray.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-02
"""

import ray
import gym
import copy
from typing import List, Dict, Callable
from tqdm import tqdm

from mrl_analysis.utility.interfaces import MetaRLPolicy

class MdpPathCollector():
    """A path collector which is able to collect a specific number of paths.

    Parameters
    ----------
    env : gym.Env
        Environment for rollouts
    policy : Policy
        Policy for rollouts
    rollout_fn : Callable
        Rollout function 
        (see e.g. rlkit/sampler/rollout_functions/rollout or
        smrl/data_management/rollout_functions/_rollout_with_encoder)
    """
    def __init__(
        self,
        env: gym.Env,
        policy: MetaRLPolicy,
        rollout_fn: Callable,
        *args,
        **kwargs,
    ) -> None:
        self._rollout_fn = rollout_fn
        self._policy = policy
        self._env = env
        self._render = kwargs.get('render', False)
        self._render_kwargs = kwargs.get('render_kwargs', {})

    def collect_new_paths(
        self,
        max_path_length: int,
        num_paths: int,
        *args,
        **kwargs,
    ) -> List[Dict]:
        """Return ``num_paths`` new paths

        Parameters
        ----------
        max_path_length : int
            Maximum length of a single path
        num_paths : int
            Number of paths to collect

        Returns
        -------
        List[Dict]
            List of path dictionaries.
        """
        paths = []
        for _ in tqdm(range(num_paths)):
            path = self._rollout_fn(
                self._env,
                self._policy,
                max_path_length=max_path_length,
                render=self._render,
                render_kwargs=self._render_kwargs,
            )
            paths.append(path)
        return paths


@ray.remote
def _rollout_thread(rollout_fn: Callable, *args, **kwargs):
    """Wrapper for rollout functions to run them without blocking (multithreaded).

    Use _rollout_thread.remote() to start the thread and ray.get() to obtain results.

    Parameters
    ----------
    rollout_fn : Callable
        Rollout function to execute

    Returns
    -------
        The results of the rollout function.
    """
    return rollout_fn(*args, **kwargs)


class MultithreadedPathCollector(MdpPathCollector):
    """A multithreaded path collector which is able to collect a specific number of paths.

    Note: Due to multithreading, this collector cannot render the sampling process

    Parameters
    ----------
    env : gym.Env
        Environment for rollouts
    policy : Policy
        Policy for rollouts
    rollout_fn : Callable
        Rollout function 
        (see e.g. rlkit/sampler/rollout_functions/rollout or
        smrl/data_management/rollout_functions/_rollout_with_encoder)
    """
    def __init__(self, env: gym.Env, policy: MetaRLPolicy, rollout_fn: Callable, *args, **kwargs):
        if not ray.is_initialized():
            ray.init(num_gpus=0)
        super().__init__(env, policy, rollout_fn, *args, **kwargs)
        # Multithreading preparation
        self._render = False

    def collect_new_paths(
        self,
        max_path_length: int,
        num_paths: int,
        *args,
        **kwargs,
    ) -> List[Dict]:
        path_tasks = []
        for _ in range(num_paths):
            self._env.reset()
            path_tasks.append(_rollout_thread.remote(
                self._rollout_fn,
                copy.deepcopy(self._env),
                self._policy,
                max_path_length=max_path_length,
                render=False,
            ))
        paths = []
        with tqdm(total=num_paths) as pbar:
            while len(path_tasks) > 0:
                ready, path_tasks = ray.wait(path_tasks)
                paths.extend(copy.deepcopy(ray.get(ready))) 
                # Copying is (admittedly) not elegant but it helps to avoid 
                # objects which are pinned in memory by ray for MuJoCo environments.
                # These could eventually lead to memory leakage because they
                # were accumulated over time.
                # You might also want to look at https://github.com/ray-project/ray/issues/9504
                # for a similar issue.
                pbar.update(len(ready))
        return paths