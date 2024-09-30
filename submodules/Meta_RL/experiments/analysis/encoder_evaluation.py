"""
This file contains the function ``encoder_plots()`` which visualizes training 
progress data and latent spaces of trained encoders. For training, the function
assumes that the ``EncoderAlgorithm`` class has been used.

Refer to model_evaluation.py for visualizing training progress of ``MetaRlAlgorithm``
instances instead.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-04-06
"""

import json
import os
from pathlib import Path
from pprint import pprint
from typing import Tuple
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from mrl_analysis.trajectory_rollout.trajectory import TrajectoryGenerator
from mrl_analysis.trajectory_rollout.encoding import encodings_from_encoder
from mrl_analysis.plots.latent_space import plot_latent_space, plot_latent_space_axis_over_task
from mrl_analysis.utility.data_smoothing import smooth_plot, smooth_fill_between
from smrl.utility.ops import ensure_importable_entries, CustomJsonEncoder
from smrl.experiment.encoder_training import init_models
from smrl.utility.console_strings import bold, italic
from smrl.policies.exploration import RandomMemoryPolicy
from smrl.policies.base import Policy

import matplotlib
matplotlib.use("Agg")


def encoder_plots(
    encoder_path: str, 
    save_dir: str, 
    test_policy: Policy = None,
    env: gym.Env = None,
    figure_size: Tuple[int,int] = (20, 10),
    multithreading: bool = True,
    n_trajectories: int = 50,
    len_trajectories: int = 250,
    color_by: str = 'rewards',
    specification_keyword: str = 'goal',  # How tasks are named
    show_plots: bool = False,
):
    """
    Loads and plots encoder training data:
    - Progress plots (ELBO, prediction error)
    - Latent space plots
    - Movements
    """

    os.environ['MULTITHREADING'] = "True" if multithreading else "False"

    def save_figure(fig: Figure, save_as: Path):
        # fig.set_size_inches(*figure_size)
        fig.tight_layout()
        fig.savefig(save_as.with_suffix('.svg'))
        fig.savefig(save_as.with_suffix('.png'))
        fig.savefig(save_as.with_suffix('.pdf'))

    """ Load data """

    with open(os.path.join(encoder_path, "variant.json"), "r") as file:
        config = json.load(file)
    with open(os.path.join(encoder_path, "progress.csv"), "r") as file:
        progress = pd.read_csv(file)

    config = ensure_importable_entries(config)
    name = Path(encoder_path).name

    print("\nEvaluation for " + bold(name))

    env, _ = config['environment_factory']()
    encoder, decoder, vae, policy = init_models(config)
    if test_policy is not None:
        policy = test_policy
    encoder.train(False)
    policy.train(False)

    params = torch.load(os.path.join(encoder_path, "params.pkl"), map_location='cpu')
    encoder.load_state_dict(params['encoder'])

    dir = Path(save_dir).joinpath(name)
    dir.mkdir(exist_ok=True, parents=True)    

    # Save config file to directory
    with open(dir.joinpath("config.json"), mode="w") as file:
        json.dump(config, file, indent=4, cls=CustomJsonEncoder)


    """ Training curves """

    x = progress['num train calls']

    fig, axs = plt.subplots(1, figsize=figure_size)
    smooth_plot(axs, x, progress['elbo'])
    axs.set_xlabel('Inference train calls')
    axs.legend(['ELBO'])
    save_figure(fig, dir.joinpath("elbo"))

    fig, axs = plt.subplots(1, figsize=figure_size)
    smooth_plot(axs, x, progress['reward prediction error'], label="Reward prediction error")
    smooth_plot(axs, x, progress['state prediction error'], label="State prediction error")
    axs.set_xlabel('Inference train calls')
    axs.set_yscale('log')
    axs.legend()
    save_figure(fig, dir.joinpath("prediction_errors"))


    """ Latent space plots """

    tg = TrajectoryGenerator(env, policy, encoder)

    trajectories = tg.run(n_trajectories, len_trajectories)

    encodings, goals, color_information = [], [], []
    for traj in trajectories:
        for i in range(encoder.context_size, len(traj['observations']), 5):
            encodings.append(traj['encodings'][i][None])
            goals.append(traj['tasks'][i][specification_keyword][None, ...])
            color_information.append(traj['contexts'][color_by][i, -1])
    encodings = np.concatenate(encodings)
    goals = np.concatenate(goals)
    color_information = np.concatenate(color_information)

    # Latent space figures
    print("Latent space figures ...")
    fig, axs = plot_latent_space(encodings[:,:2], goals, size=figure_size[1])
    save_figure(fig, dir.joinpath("latent-space_time-ordered"))

    fig, axs = plot_latent_space_axis_over_task(encodings[:,:2], goals, color_information)
    fig.set_size_inches(*figure_size)
    save_figure(fig, dir.joinpath("tasks-and-encodings_time-ordered"))

    # Movements
    fig, axs = plt.subplots(2, figsize=(figure_size[0], 2*figure_size[1]))
    dim = 0
    axs[0].set_title("Movements")
    axs[1].set_title("Encodings")
    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Encoding dist.")
    axs[0].set_ylabel(f"Position, dim {dim}")
    for i, traj in enumerate(trajectories[::10]):
        tasks = [task[specification_keyword][dim] for task in traj['tasks']]
        axs[0].plot(range(len(traj['observations'][:,dim])), traj['observations'][:,dim], c=f"C{i}")
        axs[0].plot(range(len(traj['observations'][:,dim])), tasks, c=f"C{i}", linestyle='dashed')
        axs[1].fill_between(range(len(traj['latent_mean'])), (traj['latent_mean'] + traj['latent_std'])[:,0], (traj['latent_mean'] - traj['latent_std'])[:,0], color=f"C{i}", alpha=0.2)
        axs[1].plot(range(len(traj['latent_mean'])), traj['latent_mean'][:,0], c=f"C{i}")
    save_figure(fig, dir.joinpath("trajectory-1d"))

    # Latent space, randomized
    print("Collecting encodings ...")
    encodings, tasks, contexts, trajectories = encodings_from_encoder(
        encoder, policy, env, n_trajectories, len_trajectories, 
        encodings_per_trajectory=25, randomize_samples=True,
    )
    goals = np.array([task[specification_keyword] for task in tasks])
    color_information = contexts[color_by][:,-1,0]

    # Latent space figures
    print("Latent space figures ...")
    fig, axs = plot_latent_space(encodings[:,:2], goals, size=figure_size[1])
    save_figure(fig, dir.joinpath("latent-space_randomized"))

    fig, axs = plot_latent_space_axis_over_task(encodings[:,:2], goals, color_information)
    fig.set_size_inches(*figure_size)
    save_figure(fig, dir.joinpath("tasks-and-encodings_randomized"))

    print(italic("Done!"))

    if show_plots:
        plt.show()

    plt.close('all')



if __name__ == "__main__":
    paths = [
        # "data/experiments/encoder_training/toy1d/attention-encoder-np",
        # "data/experiments/encoder_training/toy1d/attention-encoder-vae",
        # "data/experiments/encoder_training/toy1d/attention-encoder-vae-no-self-attention",
        # "data/experiments/encoder_training/toy1d/gru-encoder-np",
        # "data/experiments/encoder_training/toy1d/gru-encoder-vae",
        # "data/experiments/encoder_training/toy1d/mlp-encoder-vae",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline with multiplier_2023-11-18_19-59-28",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline with multiplier_2023-11-18_19-59-28_transfer",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_2023-11-18_19-42-43",
        "/home/ubuntu/juan/Meta-RL/data/Baseline with multiplier_2023-11-18_19-59-28_transfer"

    ]

    # policy = RandomMemoryPolicy(1, 10, 0.0, 0.0001, 0.0001)
    policy = None

    for path in paths:
        encoder_plots(path, save_dir = "./evaluation/encoder_evaluation", test_policy=policy)
