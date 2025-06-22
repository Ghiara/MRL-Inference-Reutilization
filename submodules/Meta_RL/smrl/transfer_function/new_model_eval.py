"""
This file contain shte function ``model_evaluation()`` which visualizes
the training progress of trained agents as well of their final models - both
in terms of exemplary trajectories and latent space plots.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-04-06
"""


# from transfer_configs.half_cheetah_config import config

# from experiments_configs.walker_multi import config as walker_config

import os
from typing import Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import gym
import numpy as np
import pandas as pd
import argparse

from smrl.experiment.analysis import load_results
from smrl.utility.console_strings import bold, italic
from smrl.utility.ops import CustomJsonEncoder
# from smrl.policies.exploration import RandomMemoryPolicy, RandomPolicy

from mrl_analysis.utility.interfaces import MdpEncoder, MetaRLPolicy, MetaQFunction
from mrl_analysis.trajectory_rollout.encoding import encodings_from_encoder
from mrl_analysis.plots.latent_space import plot_latent_space, plot_latent_space_axis_over_task
from mrl_analysis.video.video_creator import VideoCreator
from mrl_analysis.utility.data_smoothing import smooth_plot, smooth_fill_between

from main_config import HARDCODED
# from smrl.policies.meta_policy import PretrainedCheetah
from smrl.vae.encoder_decorators.io_modification import InputOutputDecorator
# from meta_envs.mujoco.cheetah import HalfCheetahEnvExternalTask
from meta_envs.toy_goal.toy_1d import Toy1D
from specific.encoder_transfer import map_cheetah_to_toy1d
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from sac_envs.walker_multi import WalkerMulti
from sac_envs.hopper_multi import HopperMulti
from smrl.transfer_function.new_PathCollector import TrajectoryGeneratorWithTransferFunction
from collections import OrderedDict
# from smrl.trainers.transfer_function import TransferFunction
from model import PolicyNetwork as TransferFunction

from mrl_analysis.plots.plot_settings import *

# HARDCODED = True

# Parameters
video_creator = VideoCreator()
video_creator.fps = 30
DEVICE = 'cpu'
max_path_l = 200

def get_complex_agent(env, complex_agent_config):
    pretrained = os.path.join(complex_agent_config['path'],"/models/policy_model/epoch_",str(complex_agent_config['epoch']),".pth")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]
    transfer_function = TransferFunction(
        n_states=n_states,
        n_actions=n_actions,
        action_bounds=action_bounds,
        pretrained=pretrained
        )
    transfer_function.to(DEVICE)
    return transfer_function

def model_evaluation(
    path_to_data: str, 
    save_dir: str, 
    config,
    itr: int = None, 
    env: gym.Env = None,
    policy: MetaRLPolicy = None,
    figure_size: Tuple[int,int] = (20, 10),
    trajectory_1d: bool = True,
    trajectory_2d: bool = False,
    multithreading = True,
    n_trajectories: int = 1,
    color_by: str = 'rewards',
    specification_keyword: str = 'goal',  # How tasks are named
    create_video: bool = True,
    show_plots: bool = False,
    encoder_path: str = None,
    agent_path: str = None,
    epoch: int = 1,
):
    """Creates evaluation figures from a directory of training data.

    Parameters
    ----------
    path_to_data : str
        Training directory. Must contain
            Weights: 'params.pkl' or 'itr_<nr.>.pkl'
            Progress file: 'progress.csv'
            Config file: 'variant.json'
    save_dir : str
        Directory into which plots are saved
    itr : int, optional
        Iteration of weights, by default None
    env : gym.Env, optional
        Environment (if the environment in the configuration file should 
        not be used), by default None
    policy : MetaRLPolicy, optional
        Policy for rollouts (if the trained policy should not be used),
        by default None
    figure_size : Tuple[int,int], optional
        Size of the saved figures, by default (20, 10)
    trajectory_1d : bool, optional
        Set to True to create 1d-trajectory plots (first dimension), by default True
    trajectory_2d : bool, optional
        Set to True to create 2d-trajectory plots (first two dimensions), by default False
    multithreading : bool, optional
        Set to True to use ray for rollouts, by default True
    n_trajectories : int, optional
        Number of trajectories for evaluation, by default 50
    color_by : str, optional
        Color coding for task-encoding plots, choose one of
        |'observations'|'rewards'|'actions'|'next_observations'|'terminals'|, 
        by default 'rewards'
    specification_keyword : str, optional
        Keyword in task dictionaries (dependent on environment) which belongs
        to the task feature, by default 'goal'
    show_plots : bool, optional
        Set to True to print plots to screen, by default False
    """
    
    def save_figure(fig: Figure, save_as: Path):
        # fig.set_size_inches(*figure_size)
        fig.tight_layout()
        fig.savefig(save_as.with_suffix('.svg'))
        fig.savefig(save_as.with_suffix('.png'))
        fig.savefig(save_as.with_suffix('.pdf'))

    # os.environ['MULTITHREADING'] = "True" if multithreading else "False"

    # Load data
    path_to_data = Path(path_to_data)
    result_dict = load_results(path_to_data, itr=itr)
    name = path_to_data.name
    progress: pd.DataFrame = result_dict['progress']
    max_path_length = result_dict['config']['algorithm_kwargs']['max_path_length']
    max_path_length = max_path_l

    # Initialize directory for storing images
    dir = Path(save_dir).joinpath(name)
    dir.mkdir(exist_ok=True, parents=True)

    print(bold("\nEvaluation for " + italic(name) + f", Epoch {result_dict['epoch']}"))

    # Save config file to directory
    with open(dir.joinpath("config.json"), mode="w") as file:
        json.dump(result_dict['config'], file, indent=4, cls=CustomJsonEncoder)
    progress.to_csv(dir.joinpath("progress.csv"), sep=",")

    # Figures

    # Policy training
    try:
        print("Policy training plots ...")

        x = progress['trainer/Policy trainer/num train calls']

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_fill_between(axs, x, progress['eval/Returns Min'], progress['eval/Returns Max'], color='lightgrey', alpha=0.3)
        smooth_plot(axs, x, progress['eval/Average Returns'], color=c_avg, label='Average returns')
        smooth_plot(axs, x, progress['eval/Returns Max'], color=c_max, label='Maximum returns', linestyle='dashed', plot_original=False)
        smooth_plot(axs, x, progress['eval/Returns Min'], color=c_min, label='Minimum returns', linestyle='dashed', plot_original=False)
        axs.set_xlabel('Train steps (policy)')
        axs.legend()
        save_figure(fig, dir.joinpath("returns"))

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Policy trainer/Log Pis Mean'], label="Mean logprobs")
        smooth_plot(axs, x, progress['trainer/Policy trainer/Log Pis Min'], label="Minimum logprobs")
        smooth_plot(axs, x, progress['trainer/Policy trainer/Log Pis Max'], label="Maximum logprobs")
        axs.legend()
        axs.set_xlabel('Train steps (policy)')
        save_figure(fig, dir.joinpath("Action logprobs"))

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Policy trainer/policy/normal/std Mean'], label="Standard deviation (mean)")
        smooth_plot(axs, x, progress['trainer/Policy trainer/policy/normal/std Min'], label="Standard deviation (min)")
        smooth_plot(axs, x, progress['trainer/Policy trainer/policy/normal/std Max'], label="Standard deviation (max)")
        axs.set_xlabel('Train steps (policy)')
        axs.legend()
        save_figure(fig, dir.joinpath("Action variation"))

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Policy trainer/Q Targets Mean'], label="Mean Q")
        smooth_fill_between(axs, x, 
                        progress['trainer/Policy trainer/Q Targets Mean'] - progress['trainer/Policy trainer/Q Targets Std'],
                        progress['trainer/Policy trainer/Q Targets Mean'] + progress['trainer/Policy trainer/Q Targets Std'],
                        alpha=0.3,
                        color=c_avg,
                        )
        smooth_plot(axs, x, progress['trainer/Policy trainer/Q Targets Min'], label="Min Q", color=c_min)
        smooth_plot(axs, x, progress['trainer/Policy trainer/Q Targets Max'], label="Max Q", color=c_max)
        axs.set_xlabel('Train steps (policy)')
        axs.legend()
        save_figure(fig, dir.joinpath("Q-target values"))

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Policy trainer/QF1 Loss'], label="QF1 loss")
        smooth_plot(axs, x, progress['trainer/Policy trainer/QF2 Loss'], label="QF2 loss")
        axs.set_xlabel('Train steps (policy)')
        axs.legend()
        axs.set_yscale('log')
        save_figure(fig, dir.joinpath("qfunction_loss"))

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Policy trainer/Policy Loss'], label="Policy loss")
        axs.set_xlabel('Train steps (policy)')
        axs.legend()
        save_figure(fig, dir.joinpath("policy_loss"))
    except KeyError:
        print("(skipped)")


    # Inference training
    try:
        print("Inference training plots ...")
        x = progress['trainer/Inference trainer/num train calls']

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Inference trainer/elbo'])
        axs.set_xlabel('Train steps (inference)')
        axs.legend(['ELBO'])
        save_figure(fig, dir.joinpath("elbo"))

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Inference trainer/reward prediction error'], label="Reward prediction error")
        smooth_plot(axs, x, progress['trainer/Inference trainer/state prediction error'], label="State prediction error")
        axs.set_xlabel('Train steps (inference)')
        axs.set_yscale('log')
        axs.legend()
        save_figure(fig, dir.joinpath("prediction_errors"))
    except KeyError:
        print("(skipped)")


    # Load networks
    encoder: MdpEncoder = result_dict['encoder']
    decoder: MdpEncoder = result_dict['decoder']
    if env is None:
        env = result_dict['eval_env']
    
    if HARDCODED == True:
        # result_dict['eval_policy'] = RandomPolicy(action_dim=1, std=torch.tensor(0.5))
        result_dict['eval_policy'] = PretrainedCheetah()
    if policy is None:
        policy = result_dict['eval_policy']

    value_function: MetaQFunction = result_dict['qf1']

    encoder.train(False)
    decoder.train(False)
    policy.train(False)
    value_function.train(False)


    # Rollout trajectories    
    simple_env = Toy1D(25,25, min_pos=.0, max_pos=10.0, max_action=0.5)
    encoder, env, transfer_function = load_model(encoder, encoder_path, agent_path, epoch)
    decoder = None
    value_function = None
    tg = TrajectoryGeneratorWithTransferFunction(simple_env, env, policy, encoder, transfer_function, decoder, value_function)
    # tg = TrajectoryGenerator(env, policy, encoder, None, value_function)
    trajectories = tg.run(n_trajectories, max_path_length)

    encodings, goals, color_information = [], [], []
    for traj in trajectories:
        for i in range(encoder.context_size, len(traj['observations']), 5):
            encodings.append(traj['encodings'][i][None])
            goals.append(traj['tasks'][i][specification_keyword][None, ...])
            if color_by is not None:
                color_information.append(traj['contexts'][color_by][i, -1])
    encodings = np.concatenate(encodings)
    goals = np.concatenate(goals)
    if color_by is not None:
        color_information = np.concatenate(color_information)
    else:
        color_information = None

    # Save trajectories to csv files
    dir.joinpath("trajectories/").mkdir(parents=True, exist_ok=True)
    for i, traj in enumerate(trajectories):
        for k, v in traj['tasks'][0].items():
            if isinstance(v, np.ndarray):
                traj['task/'+k] = np.vstack([task[k] for task in traj['tasks']])
            else:
                traj['task/'+k] = np.array([task[k] for task in traj['tasks']])
            if traj['task/'+k].ndim < 2:
                traj['task/'+k] = traj['task/'+k][:,None]
        keys, values = [], []
        for key, value in traj.items():
            if isinstance(value, np.ndarray):
                if value.ndim < 2:
                    continue
                keys.extend([key + "." + str(i) for i in range(value.shape[1])])
                values.append(value)

        header = "".join([k + "," for k in keys])[:-1]
        values = np.hstack(values)
        np.savetxt(dir.joinpath("trajectories").joinpath(f"{i+1}.csv"), values, header=header, delimiter=",", comments="")


    # Latent space figures
    print("Latent space figures ...")
    fig, axs = plot_latent_space(encodings[:,:2], goals, size=figure_size[1])
    save_figure(fig, dir.joinpath("latent-space_time-ordered"))

    fig, axs = plot_latent_space_axis_over_task(encodings[:,:2], goals, color_information)
    fig.set_size_inches(*figure_size)
    save_figure(fig, dir.joinpath("tasks-and-encodings_time-ordered"))


    if create_video:
        print("Video ...")
        video_creator.create_video(
            result_dict, video_length=20.0, 
            save_as=str(dir.joinpath("video.mp4")), 
            env_reset_interval=max_path_length, 
            width=env.screen_width, height=env.screen_height,
        )

    # Movements
    if trajectory_1d:
        fig, axs = plt.subplots(2, figsize=(figure_size[0], 2*figure_size[1]))
        dim = 0
        axs[0].set_title("Position")
        axs[1].set_title("Latent distribution (mean and variance)")
        # axs[2].set_title("Q-function value")
        # axs[3].set_ylabel("Action distribution")
        # axs[3].set_title("Actions")
        for a in axs:
            a.set_xlabel("Time (in steps)")
        for i, traj in enumerate(trajectories[::10]):
            tasks = [task[specification_keyword][dim] for task in traj['tasks']]
            axs[0].plot(range(len(traj['observations'][:,-3])), traj['observations'][:,-3], c=f"C{i}")
            axs[0].plot(range(len(traj['observations'][:,-3])), tasks, c=f"C{i}", linestyle='dashed', alpha=0.5)
            axs[1].fill_between(range(len(traj['latent_mean'])), (traj['latent_mean'] + traj['latent_std'])[:,0], (traj['latent_mean'] - traj['latent_std'])[:,0], color=f"C{i}", alpha=0.2)
            axs[1].plot(range(len(traj['latent_mean'])), traj['latent_mean'][:,0], c=f"C{i}")
            # axs[2].plot(range(len(traj['qvalues'])), traj['qvalues'], c=f"C{i}")
            # axs[3].fill_between(range(len(traj['actions'])), (traj['action_mean'] - traj['action_std'])[:,0], (traj['action_mean'] + traj['action_std'])[:,0], alpha=0.3, color=f"C{i}")
            # axs[3].plot(range(len(traj['actions'])), traj['action_mean'], c=f"C{i}")
            # axs[3].plot(range(len(traj['actions'])), traj['actions'], c=f"C{i}")
        save_figure(fig, dir.joinpath("trajectory-1d"))
    if trajectory_2d:
        fig, axs = plt.subplots(1, 2, figsize=(2*figure_size[1], figure_size[1]))
        dim = 0
        axs[0].set_title("Trajectory")
        axs[1].set_title("Encodings")
        for i, traj in enumerate(trajectories[::10]):
            tasks = [np.atleast_2d(task[specification_keyword]) for task in traj['tasks']]
            tasks = np.concatenate(tasks)
            axs[0].plot(traj['observations'][:,0], traj['observations'][:,1], c=f"C{i}", alpha=0.7)
            axs[0].scatter(tasks[:,0], tasks[:,1], c=f"C{i}")
            axs[1].plot(traj['encodings'][:,0], traj['encodings'][:,1], c=f"C{i}", alpha=0.7)
        save_figure(fig, dir.joinpath("trajectory-2d"))

    # Latent space, randomized
    print("Collecting encodings ...")
    encodings, tasks, contexts, trajectories = encodings_from_encoder(
        encoder, policy, env, n_trajectories, max_path_length, 
        encodings_per_trajectory=25, randomize_samples=True,
    )
    goals = np.array([task[specification_keyword] for task in tasks])
    if color_by is not None:
        color_information = contexts[color_by][:,-1,0]
    else:
        color_information = None

    # Latent space figures
    print("Latent space figures ...")
    fig, axs = plot_latent_space(encodings[:,:2], goals, size=figure_size[1])
    save_figure(fig, dir.joinpath("latent-space_randomized"))

    fig, axs = plot_latent_space_axis_over_task(encodings[:,:2], goals, color_information)
    fig.set_size_inches(*figure_size)
    save_figure(fig, dir.joinpath("tasks-and-encodings_randomized"))


    if show_plots: plt.show()
    plt.close('all')    # Reduce memory requirements by closing open plots

    # Video

    print(italic("Done!"))
    print("\n\n")


def load_model(encoder, encoder_path, agent_path, epoch):

    '''
    Define the low-level policy and agent to test
    '''
    complex_agent_config = dict(
        path = agent_path,
        epoch = epoch,
    )
    with open(os.path.join(agent_path, 'config.json'), "r") as file:
        env_config = json.load(file)

    complex_agent_config['environment'] = WalkerMulti(env_config)
    env = complex_agent_config['environment']

    encoder = InputOutputDecorator(encoder, map_cheetah_to_toy1d, observation_dim=env.observation_space, action_dim=env.action_space, latent_dim=1, encoding_dim=1, context_size=5)
    transfer_function = get_complex_agent(env, complex_agent_config)
    return encoder, env, transfer_function

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    '''
    Deinfe the inference module
    '''

    default_encoder_path = "/home/ubuntu/juan/MRL-Inference-Reutilization/data/experiments_thesis/step1_biggerNN_-10_10/_2025-06-22_21-02-48"
    default_agent_path = '/home/ubuntu/juan/MRL-Inference-Reutilization/output/low_level_policy/new_cheetah_training/half_cheetah_initial_random'

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default=None, help='Path to encoder checkpoint')
    parser.add_argument('--agent_path', type=str, default=None, help='Path to low level agent checkpoint')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch of the low level agent to use')
    args = parser.parse_args()

    encoder_path = args.encoder_path if args.encoder_path is not None else default_encoder_path
    agent_path = args.agent_path if args.agent_path is not None else default_agent_path
    epoch = args.epoch if args.epoch is not None else 100


    with open(os.path.join(agent_path, 'variant.json'), "r") as file:
        config = json.load(file)
    model_evaluation(
        agent_path,
        save_dir='./evaluation/experiments_thesis/transfer_full_walker_5_test/',
        config=config,
        # save_dir = './data/delete',
        create_video=True,
        figure_size=(8,6),
        trajectory_2d=False,
        color_by=None,
        encoder_path=encoder_path, 
        agent_path=agent_path,
        epoch=epoch,
    )