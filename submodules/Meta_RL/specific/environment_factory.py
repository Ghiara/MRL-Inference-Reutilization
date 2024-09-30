"""
This file provides specific environment factory functions for scaled versions
of the one-dimensional toy environment.
Additionally, it implements translation functions for an encoder transfer.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-03-13
"""

from typing import Tuple

from meta_envs.toy_goal import Toy1D
from smrl.utility.console_strings import print_to_terminal



def toy1d_rand_smaller() -> Tuple[Toy1D, Toy1D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "reset_mode": 'random',
        "max_action": 0.5,
        "min_pos": -10,
        "max_pos": 10,
    }
    expl_env = Toy1D(*env_args, **env_kwargs)
    eval_env = Toy1D(*env_args, **env_kwargs)
    return expl_env, eval_env

def input_map_smaller(obs, act, rew, n_obs, term):
    return 2*obs, 2*act, 2*rew, 2*n_obs, term

def toy1d_rand_larger() -> Tuple[Toy1D, Toy1D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "reset_mode": 'random',
        "max_action": 0.2,
        "min_pos": -2.0,
        "max_pos": 2.0,
    }
    expl_env = Toy1D(*env_args, **env_kwargs)
    eval_env = Toy1D(*env_args, **env_kwargs)
    return expl_env, eval_env

def input_map_larger(obs, act, rew, n_obs, term):
    return 0.5*obs, 0.5*act, 0.5*rew, 0.5*n_obs, term

def toy1d_rand_small() -> Tuple[Toy1D, Toy1D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "reset_mode": 'random',
        "max_action": 0.01,
        "min_pos": -0.1,
        "max_pos": 0.1,
    }
    expl_env = Toy1D(*env_args, **env_kwargs)
    eval_env = Toy1D(*env_args, **env_kwargs)
    return expl_env, eval_env

def input_map_small(obs, act, rew, n_obs, term):
    return 10*obs, 10*act, 10*rew, 10*n_obs, term

def toy1d_rand_large() -> Tuple[Toy1D, Toy1D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "reset_mode": 'random',
        "max_action": 1.0,
        "min_pos": -10.0,
        "max_pos": 10.0,
    }
    expl_env = Toy1D(*env_args, **env_kwargs)
    eval_env = Toy1D(*env_args, **env_kwargs)
    return expl_env, eval_env

def input_map_large(obs, act, rew, n_obs, term):
    return 0.10*obs, 0.10*act, 0.10*rew, 0.10*n_obs, term

def toy1d_rand_huge() -> Tuple[Toy1D, Toy1D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "reset_mode": 'random',
        "max_action": 1.0,
        "min_pos": -50.0,
        "max_pos": 50.0,
    }
    expl_env = Toy1D(*env_args, **env_kwargs)
    eval_env = Toy1D(*env_args, **env_kwargs)
    return expl_env, eval_env

def input_map_huge(obs, act, rew, n_obs, term):
    return 0.02*obs, 0.02*act, 0.02*rew, 0.02*n_obs, term