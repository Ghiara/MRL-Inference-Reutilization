from typing import Tuple
import numpy as np

from meta_envs.toy_goal import Toy1D, Toy1dContinuous, Toy2D
from meta_envs.wrappers import NoisyEnv, DomainRandomizer
from meta_envs.mujoco.cheetah import HalfCheetahGoal, HalfCheetahVel
from meta_envs.mujoco.ant import AntGoal, AntVel
from meta_envs.toy_goal.toy_goal_legacy import ToyGoalEnv


def toy1d() -> Tuple[Toy1D, Toy1D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 1000,
        "task_generation_mode": 'random',
        "max_action": 0.01,
        "min_pos": .1,
        "max_pos": 10.0,
    }
    expl_env = Toy1D(*env_args, **env_kwargs)
    eval_env = Toy1D(*env_args, **env_kwargs)
    return expl_env, eval_env

def toy1d_rand() -> Tuple[Toy1D, Toy1D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "reset_mode": 'random',
        "max_action": 3.0,
        "min_pos": -10.0,
        "max_pos": 10.0,
    }
    expl_env = Toy1D(*env_args, **env_kwargs)
    eval_env = Toy1D(*env_args, **env_kwargs)
    return expl_env, eval_env

def toy1d_domain_rand(multiplier:Tuple) -> Tuple[Toy1D, Toy1D]:
    expl_env, eval_env = toy1d_rand()
    #  max_multiplier = 2
    # min_multiplier = 0.5
    return DomainRandomizer(expl_env, multiplier[0], multiplier[1]), DomainRandomizer(eval_env, multiplier[0], multiplier[1])

def toy1d_rand_noisy() -> Tuple[Toy1D, Toy1D]:
    expl_env, eval_env = toy1d_rand()
    noise_levels = (
        0.0,    # actions
        0.005,  # observations
        0.05    # rewards
    )
    return NoisyEnv(expl_env, noise_levels), NoisyEnv(eval_env, noise_levels)

def toy1d_without_boundary() -> Tuple[Toy1D, Toy1D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "min_pos": -np.inf,
        "max_pos": np.inf,
        "max_action": 2.0,
        "task_scale": 25.0,
        "reset_mode": "random",
    }
    expl_env = Toy1D(*env_args, **env_kwargs)
    eval_env = Toy1D(*env_args, **env_kwargs)
    return expl_env, eval_env

def toy1d_without_boundary_zero() -> Tuple[Toy1D, Toy1D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "min_pos": -np.inf,
        "max_pos": np.inf,
        "max_action": 2.0,
        "task_scale": 25.0,
        "reset_mode": "zero",
    }
    expl_env = Toy1D(*env_args, **env_kwargs)
    eval_env = Toy1D(*env_args, **env_kwargs)
    return expl_env, eval_env

def small_toy1d_without_boundary() -> Tuple[Toy1D, Toy1D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "min_pos": -np.inf,
        "max_pos": np.inf,
        "max_action": 1.0,
        "task_scale": 1.0,
        "reset_mode": "random",
    }
    expl_env = Toy1D(*env_args, **env_kwargs)
    eval_env = Toy1D(*env_args, **env_kwargs)
    return expl_env, eval_env

def toy1d_continuous() -> Tuple[Toy1dContinuous, Toy1dContinuous]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "mass": 1.0,
        "spring_constant": 0.0,
        "damping": 0.5,
    }
    expl_env = Toy1dContinuous(*env_args, **env_kwargs)
    eval_env = Toy1dContinuous(*env_args, **env_kwargs)
    return expl_env, eval_env

def toy1d_cont_rand() -> Tuple[Toy1dContinuous, Toy1dContinuous]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "mass": 1.0,
        "spring_constant": 0.0,
        "damping": 0.5,
        "reset_mode": "random",
    }
    expl_env = Toy1dContinuous(*env_args, **env_kwargs)
    eval_env = Toy1dContinuous(*env_args, **env_kwargs)
    return expl_env, eval_env

toy1d_discretized = toy1d_continuous    # Legacy support

def toy2d() -> Tuple[Toy2D, Toy2D]:
    return toy2d_L2()

def toy2d_rand() -> Tuple[Toy2D, Toy2D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "reward_type": "L2",
        "reset_mode": "random",
    }
    expl_env = Toy2D(*env_args, **env_kwargs)
    eval_env = Toy2D(*env_args, **env_kwargs)
    return expl_env, eval_env

def toy2d_L1() -> Tuple[Toy2D, Toy2D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "reward_type": "L1",
    }
    expl_env = Toy2D(*env_args, **env_kwargs)
    eval_env = Toy2D(*env_args, **env_kwargs)
    return expl_env, eval_env

def toy2d_L2() -> Tuple[Toy2D, Toy2D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "reward_type": "L2",
    }
    expl_env = Toy2D(*env_args, **env_kwargs)
    eval_env = Toy2D(*env_args, **env_kwargs)
    return expl_env, eval_env

def cheetah_vel() -> Tuple[HalfCheetahVel, HalfCheetahVel]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_prob": 0,
        "render_mode": 'rgb_array',
    }
    expl_env = HalfCheetahVel(*env_args, **env_kwargs)
    eval_env = HalfCheetahVel(*env_args, **env_kwargs)
    return expl_env, eval_env

def cheetah_goal() -> Tuple[HalfCheetahGoal, HalfCheetahGoal]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_prob": 0,
        "render_mode": 'rgb_array',
    }
    expl_env = HalfCheetahGoal(*env_args, **env_kwargs)
    eval_env = HalfCheetahGoal(*env_args, **env_kwargs)
    return expl_env, eval_env

def cheetah_goal_onesided() -> Tuple[HalfCheetahGoal, HalfCheetahGoal]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_prob": 0,
        "render_mode": 'rgb_array',
        "one_sided_tasks": True,
    }
    expl_env = HalfCheetahGoal(*env_args, **env_kwargs)
    eval_env = HalfCheetahGoal(*env_args, **env_kwargs)
    return expl_env, eval_env

def ant_goal() -> Tuple[AntGoal, AntGoal]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_prob": 0,
        "render_mode": 'rgb_array',
    }
    expl_env = AntGoal(*env_args, **env_kwargs)
    eval_env = AntGoal(*env_args, **env_kwargs)
    return expl_env, eval_env

def ant_goal_onesided() -> Tuple[AntGoal, AntGoal]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "render_mode": 'rgb_array',
        "one_sided_tasks": True,
    }
    expl_env = AntGoal(*env_args, **env_kwargs)
    eval_env = AntGoal(*env_args, **env_kwargs)
    return expl_env, eval_env

def ant_vel() -> Tuple[AntGoal, AntGoal]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "render_mode": 'rgb_array',
    }
    expl_env = AntVel(*env_args, **env_kwargs)
    eval_env = AntVel(*env_args, **env_kwargs)
    return expl_env, eval_env


def old_toy_goal() -> Tuple[ToyGoalEnv, ToyGoalEnv]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 40,
        "n_eval_tasks": 30,
        "n_grid_tasks": 8,
        "use_normalized_env": False,
        "state_reconstruction_clip": 12,
        "use_state_decoder": False,
        "change_steps": 1000,
        "grid_mode": "none",
        "step_size": 0.5,
        "task_max_radius": 25.0,
	    "task_goal_offset": 0.0,
        "goal_radius": 0.2,
        "goal_1d": True,
        "one_side_goals": False
    }
    expl_env = ToyGoalEnv(*env_args, **env_kwargs)
    eval_env = ToyGoalEnv(*env_args, **env_kwargs)
    return expl_env, eval_env


def toy1d_symmetries() -> Tuple[Toy1D, Toy1D]:
    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
    }
    expl_env = Toy1D(*env_args, min_pos=0.0, max_pos=1.0, **env_kwargs)
    eval_env = Toy1D(*env_args, min_pos=-1.0, max_pos=1.0, **env_kwargs)
    return expl_env, eval_env
