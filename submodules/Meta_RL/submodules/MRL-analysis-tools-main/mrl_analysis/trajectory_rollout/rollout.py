"""
This module contains functions for policy rollouts.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2022-12-12

Note:
    The functions in this module serve the same functionality as those of
    `rlkit/samplers/rollout_functions.py`. See https://github.com/rail-berkeley/rlkit.
"""

import numpy as np
import torch
import copy
import random
from smrl.utility.console_strings import print_to_terminal
from main_config import SAVE_REWARD_PLOTS_EVALUATION


from typing import Callable, List, Tuple

from mrl_analysis.utility.interfaces import MdpEncoder, MetaRLPolicy, MetaEnv


def rollout_with_encoder(encoder: MdpEncoder, context_size: int):
    """Wrapper function which ensures that an encoder can be specified
    without breaking the argument structure of rlkit MdpPathCollector.

    The wrapped function is `_rollout_with_encoder` (see below).

    Parameters
    ----------
    encoder : Encoder
        An MDP encoder.
    context_size : int
        Length of the context sequence used for the encoder.
    """
    def rollout_fn(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
    ):
        return _rollout_with_encoder(
            env, agent, encoder, context_size,
            max_path_length, render,
            render_kwargs, preprocess_obs_for_policy_fn,
            get_action_kwargs, return_dict_obs, full_o_postprocess_func,
            reset_callback
        )
    return rollout_fn


def _rollout_with_encoder(
        env: MetaEnv,
        agent: MetaRLPolicy,
        encoder: MdpEncoder,
        context_size: int,
        max_path_length: int = np.inf,
        render: bool = False,
        render_kwargs: dict = None,
        preprocess_obs_for_policy_fn: Callable = None,
        get_action_kwargs: dict = None,
        return_dict_obs: bool = False,
        full_o_postprocess_func: Callable = None,
        reset_callback: Callable = None,
) -> dict:
    """Rollout function for Meta Reinforcement Learning. Returns one trajectory.
    An encoder is used for computing contextual information about the task.

    The path is sampled until either 
    1) The environment returns 'done' or
    2) The maximum path length (`max_path_length`) is reached.

    NOTE: This function extends the rollout function from `rlkit/samplers/rollout_functions.py`. 
    Large portions of the code are copied from rlkit.

    Parameters
    ----------
    env : gym.Env
        Environment for interaction.
    agent : MetaRLPolicy
        The agent/policy used for sampling actions.
    encoder : MdpEncoder
        The encoder used for extracting a task representation from
        contextual information.
    context_size : int
        Length of the context sequence
    max_path_length : int, optional
        Maximum length of a sampled path, by default np.inf
    render : bool, optional
        Activates (True) or deactivates (False) environment rendering,
        by default False
    render_kwargs : dict, optional
        Arguments passed to render(), will be ignored if render==False,
        by default None
    preprocess_obs_for_policy_fn : Callable, optional
        Function which preprocesses observations before
        they are passed on to the policy, by default None
    get_action_kwargs : dict, optional
        Additional arguments (besides observation) passed to
        the policy (policy.get_action), by default None
    return_dict_obs : bool, optional
        If True, the returned observations won't be mapped to a numpy array,
        by default False
    full_o_postprocess_func : Callable, optional
        Function which is called between action sampling from
        the policy and environment update (step), by default None
    reset_callback : Callable, optional
        Function which is called before sampling the path but
        after env.reset().
        Takes as inputs the environment, policy (agent), and
        current observation,
        by default None

    Returns
    -------
    dict
        Dictionary with path data:
        - "observations"
        - "encodings"
        - "actions"
        - "rewards"
        - "next_observations"
        - "terminals"
        - "dones"
        - "agent_infos"
        - "env_infos"
    """

    # Variable initialization
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x

    # TODO: If environment spaces are discrete, the shape argument might not be correct
    observations = np.zeros((max_path_length, *env.observation_space.shape))
    actions = np.zeros((max_path_length, *env.action_space.shape))
    rewards = np.zeros((max_path_length, 1))
    next_observations = np.zeros((max_path_length, *env.observation_space.shape))
    terminals = np.zeros((max_path_length, 1), dtype=np.bool8)
    dones = np.zeros((max_path_length, 1), dtype=np.bool8)
    encodings = np.zeros((max_path_length, encoder.encoding_dim))

    agent_infos = []
    env_infos = []
    tasks = []

    path_length = 0
    agent.reset()
    env.sample_task()
    o, env_info = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)

    # Sampling
    while path_length < max_path_length:
        with torch.no_grad():
            # Encode context (sample from posterior)
            index_low, index_high = max(0, path_length - context_size), path_length
            encoding = encoder.get_encoding(
                observations[index_low : index_high],
                actions[index_low : index_high],
                rewards[index_low : index_high],
                next_observations[index_low : index_high],
                terminals[index_low : index_high],
            )
            # encoding = env.task['goal'][None,0] # This line can be used for debugging purposes with Toy1D
            
            # Get action from policy
            o_for_agent = preprocess_obs_for_policy_fn(o)
            a, agent_info = agent.get_action(o_for_agent, encoding, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        # Environment step
        next_o, r, done, trunc, env_info = env.step(copy.deepcopy(a))
        if render:
            env.render(**render_kwargs)

        terminal = False
        if done:
            # terminal=False if TimeLimit caused termination
            if not env_info.pop('TimeLimit.truncated', False):
                terminal = True

        # a = a.clip(-env.max_action, env.max_action)
        a = a * env.max_action
        # Store new transition to trajectory
        observations[path_length, :] = o
        actions[path_length, :] = a
        rewards[path_length, :] = r
        next_observations[path_length, :] = next_o
        terminals[path_length, :] = terminal
        dones[path_length, :] = done
        encodings[path_length, :] = encoding

        agent_infos.append(agent_info)
        env_infos.append(env_info)
        tasks.append(env.task)
        path_length += 1

        if done:
            break

        o = next_o


    if SAVE_REWARD_PLOTS_EVALUATION:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from pathlib import Path
        from datetime import datetime
        import pytz
        import os

        current_time = datetime.now().astimezone(pytz.timezone('Europe/Berlin'))
        def save_figure(fig: Figure, save_as: Path):
            fig.savefig(save_as.with_suffix('.png'))

        directory = 'data/reward_plots/evaluation'
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Sanitize the file name and create the path
        goal_str = str(env.task['goal']).replace('[', '').replace(']', '').replace('.', ',')  # Replace/remove invalid chars
        current_time = datetime.now()
        path = f'data/reward_plots/{goal_str}_{current_time.strftime("%Y-%m-%d_%H-%M-%S")}'

        fig, axs = plt.subplots(1, figsize=(10,5))
        axs.plot(rewards, alpha=0.3)
        axs.set_xlabel('Reward')
        # axs.legend()
        save_figure(fig, Path(path))
    
    return dict(
        observations=observations[:path_length],
        encodings=encodings[:path_length],
        actions=actions[:path_length],
        rewards=rewards[:path_length],
        next_observations=next_observations[:path_length],
        terminals=terminals[:path_length],
        dones=dones[:path_length],
        agent_infos=agent_infos,
        env_infos=env_infos,
        tasks=tasks,
    )

