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

import os
from typing import Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import gym
import numpy as np
import pandas as pd
import torch

from smrl.experiment.analysis import load_results
from smrl.utility.console_strings import bold, italic
from smrl.utility.ops import CustomJsonEncoder
from smrl.policies.exploration import RandomMemoryPolicy, RandomPolicy

from mrl_analysis.utility.interfaces import MdpEncoder, MetaRLPolicy, MetaQFunction
from mrl_analysis.trajectory_rollout.trajectory import TrajectoryGenerator
from mrl_analysis.trajectory_rollout.encoding import encodings_from_encoder
from mrl_analysis.plots.latent_space import plot_latent_space, plot_latent_space_axis_over_task
from mrl_analysis.video.video_creator import VideoCreator
from mrl_analysis.utility.data_smoothing import smooth_plot, smooth_fill_between

from main_config import HARDCODED
from smrl.policies.meta_policy import PretrainedCheetah

from mrl_analysis.plots.plot_settings import *

# HARDCODED = True

# Parameters
video_creator = VideoCreator()
video_creator.fps = 30


def model_evaluation(
    path_to_data: str, 
    save_dir: str, 
    itr: int = 3000, 
    env: gym.Env = None,
    policy: MetaRLPolicy = None,
    figure_size: Tuple[int,int] = (20, 10),
    trajectory_1d: bool = True,
    trajectory_2d: bool = False,
    multithreading = True,
    n_trajectories: int = 50,
    color_by: str = 'rewards',
    specification_keyword: str = 'goal',  # How tasks are named
    create_video: bool = True,
    show_plots: bool = False,
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
    tg = TrajectoryGenerator(env, policy, encoder, decoder, value_function)
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
            axs[0].plot(range(len(traj['observations'][:,dim])), traj['observations'][:,dim], c=f"C{i}")
            axs[0].plot(range(len(traj['observations'][:,dim])), tasks, c=f"C{i}", linestyle='dashed', alpha=0.5)
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
    if create_video:
        print("Video ...")
        video_creator.create_video(
            result_dict, video_length=20.0, 
            save_as=str(dir.joinpath("video.mp4")), 
            env_reset_interval=max_path_length, 
            width=env.screen_width, height=env.screen_height,
        )

    print(italic("Done!"))
    print("\n\n")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    # Load models
    paths = [
        # "data/experiments/toy2d_on-off-policy/toy2d_rand_off-policy_log-random-inference_2023-02-26_19-58-01",
        # "data/experiments/toy2d_on-off-policy/toy2d_rand_off-policy_memory-random-inference_2023-02-26_19-57-52",
        # "data/experiments/toy2d_on-off-policy/toy2d_rand_off-policy_multi-random-inference_2023-02-26_19-57-57",
        # "data/experiments/toy2d_on-off-policy/toy2d_rand_off-policy_random-inference_2023-02-26_19-57-44",
        # "data/experiments/toy2d_on-off-policy/toy2d_rand_on-policy_2023-02-26_19-56-47",
        # "data/experiments/toy2d_on-off-policy/toy2d_off-policy_log-random-inference_2023-02-26_19-58-14",
        # "data/experiments/toy2d_on-off-policy/toy2d_rand_on-policy-entropy-tuning_2023-02-26_20-05-57",

        # "data/experiments/toy1d_buffers/toy1d_rand_context-buffer_2023-03-05_10-55-36",
        # "data/experiments/toy1d_buffers/toy1d_rand_multitask-buffer-ordered_2023-03-05_10-57-52",
        # "data/experiments/toy1d_buffers/toy1d_rand_multitask-buffer-randomized_2023-03-05_10-58-27",
        # "data/experiments/toy1d_buffers/toy1d_rand_trajectory-buffer-ordered_2023-03-05_10-56-19",
        # "data/experiments/toy1d_buffers/toy1d_rand_trajectory-buffer-randomized_2023-03-05_10-57-00",

        # "data/experiments/toy1d_on-off-policy/toy1d_rand_off-policy_log-random-inference_2023-03-15_20-20-29",
        # "data/experiments/toy1d_on-off-policy/toy1d_rand_off-policy_memory-random-inference_2023-03-17_20-26-11",
        # "data/experiments/toy1d_on-off-policy/toy1d_rand_off-policy_memory-random-inference_2023-03-19_09-52-33",
        # "data/experiments/toy1d_on-off-policy/toy1d_rand_off-policy_multi-random-inference_2023-03-16_14-11-19",
        # "data/experiments/toy1d_on-off-policy/toy1d_rand_off-policy_random-inference_2023-03-16_14-38-41",
        # "data/experiments/toy1d_on-off-policy/toy1d_rand_on-policy_2023-03-15_20-21-08",

        # "data/experiments/toy1d_TE/toy1d_TE-toy1d_2023-03-20_11-01-57",
        # "data/experiments/toy1d_TE/toy1d_rand_huge_TE-toy1d-log_2023-03-20_11-58-10",
        # "data/experiments/toy1d_TE/toy1d_rand_huge_TE-toy1d-log_2023-03-20_14-19-00",
        # "data/experiments/toy1d_TE/toy1d_rand_huge_TE-toy1d-log_2023-03-20_14-19-36",
        # "data/experiments/toy1d_TE/toy1d_rand_huge_TE-toy1d_2023-03-20_11-18-38",
        # "data/experiments/toy1d_TE/toy1d_rand_large_TE-toy1d_2023-03-20_11-02-57",
        # "data/experiments/toy1d_TE/toy1d_rand_larger_TE-toy1d_2023-03-20_11-32-57",
        # "data/experiments/toy1d_TE/toy1d_rand_small_TE-toy1d_2023-03-20_11-48-32",
        # "data/experiments/toy1d_TE/toy1d_rand_smaller_TE-toy1d_2023-03-20_11-17-34",
        # "data/experiments/toy1d_TE/toy1d_rand_huge_TE-toy1d-log-strided_2023-03-22_14-53-57",
        # "data/experiments/toy1d_TE/toy1d_rand_huge_TE-toy1d-strided_2023-03-22_13-59-59",

        # "data/experiments/toy1d_TE_mean-var/toy1d_TE-toy1d_2023-03-23_09-29-27",
        # "data/experiments/toy1d_TE_mean-var/toy1d_rand_huge_TE-toy1d-log-strided_2023-03-23_16-18-44",
        # "data/experiments/toy1d_TE_mean-var/toy1d_rand_huge_TE-toy1d-log_2023-03-23_14-27-13",
        # "data/experiments/toy1d_TE_mean-var/toy1d_rand_huge_TE-toy1d-strided_2023-03-23_15-23-37",
        # "data/experiments/toy1d_TE_mean-var/toy1d_rand_huge_TE-toy1d_2023-03-23_13-34-40",
        # "data/experiments/toy1d_TE_mean-var/toy1d_rand_large_TE-toy1d_2023-03-23_12-42-19",
        # "data/experiments/toy1d_TE_mean-var/toy1d_rand_larger_TE-toy1d_2023-03-23_10-59-49",
        # "data/experiments/toy1d_TE_mean-var/toy1d_rand_small_TE-toy1d_2023-03-23_11-50-06",
        # "data/experiments/toy1d_TE_mean-var/toy1d_rand_smaller_TE-toy1d_2023-03-23_10-14-10",
        # "data/experiments/toy1d_TE_mean-var/toy1d_rand_huge_TE-toy1d-attention-np_2023-03-25_12-01-49",
        # "data/experiments/toy1d_TE_mean-var/toy1d_rand_huge_TE-toy1d-attention-np_2023-03-25_12-14-29",
        # "data/transfer_encoders/toy1d_attention-np",

        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-on-policy_2023-03-20_10-37-15",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-on-policy-strided_2023-03-20_10-40-35",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-rand_2023-03-20_11-54-42",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-rand-strided_2023-03-20_12-03-02",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-memory_2023-03-20_13-19-32",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-memory-strided_2023-03-20_13-32-10",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-multi_2023-03-20_14-49-44",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-multi-strided_2023-03-20_15-09-16",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-log_2023-03-20_16-25-06",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-log-strided_2023-03-20_16-47-25",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-attention_2023-03-29_11-18-16",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-attention_2023-03-29_11-26-35",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-attention-strided_2023-03-29_13-07-04",

        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-log_2023-03-27_09-58-46",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-log_2023-03-27_19-05-54",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-log-strided_2023-03-27_10-00-41",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-log-strided_2023-03-27_19-06-18",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-memory_2023-03-27_14-16-18",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-memory_2023-03-28_04-41-21",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-memory-strided_2023-03-27_14-35-08",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-memory-strided_2023-03-28_04-41-57",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-multi_2023-03-27_15-41-54",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-multi_2023-03-28_07-21-03",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-multi-strided_2023-03-27_16-05-48",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-multi-strided_2023-03-28_07-19-50",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-on-policy_2023-03-27_11-23-44",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-on-policy_2023-03-27_22-39-00",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-on-policy-strided_2023-03-27_11-31-44",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-on-policy-strided_2023-03-27_22-41-54",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-rand_2023-03-27_12-51-36",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-rand_2023-03-28_01-43-27",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-rand-strided_2023-03-27_13-05-16",
        # "data/experiments/toy1d_TE_disc-to-cont_2/toy1d_cont_rand_TE-toy1d-rand-strided_2023-03-28_01-47-47",

        # "data/experiments/toy1d_encoders/toy1d_rand_attention-encoder-np_2023-03-24_09-38-33",
        # "data/experiments/toy1d_encoders/toy1d_rand_attention-encoder-vae_2023-03-24_09-38-00",
        # "data/experiments/toy1d_encoders/toy1d_rand_gru-encoder-np_2023-03-22_05-25-36",
        # "data/experiments/toy1d_encoders/toy1d_rand_gru-encoder-vae_2023-03-20_11-51-19",
        # "data/experiments/toy1d_encoders/toy1d_rand_mlp-encoder-vae_2023-03-21_09-52-45",
        # "data/experiments/toy1d_encoders/toy1d_rand_pair-encoder-np_2023-03-20_11-49-14",
        # "data/experiments/toy1d_encoders/toy1d_rand_pair-encoder-vae_2023-03-21_16-57-06",

        # "data/experiments/toy1d-cont_on-off-policy_3/toy1d_cont_rand_on-policy_2023-03-28_11-49-49",
        # "data/experiments/toy1d-cont_on-off-policy_3/toy1d_cont_rand_off-policy_random-inference_2023-03-29_09-44-05",
        # "data/experiments/toy1d-cont_on-off-policy_3/toy1d_cont_rand_off-policy_memory-random-inference_2023-03-28_11-50-37",
        # "data/experiments/toy1d-cont_on-off-policy_3/toy1d_cont_rand_off-policy_multi-random-inference_2023-03-28_10-17-55",
        # "data/experiments/toy1d-cont_on-off-policy_3/toy1d_cont_rand_off-policy_log-random-inference_2023-03-28_10-20-35",

        # "data/experiments/toy1d-cont_buffers/toy1d_cont_rand_trajectory-buffer-ordered_2023-04-23_10-53-39",
        # "data/experiments/toy1d-cont_buffers/toy1d_cont_rand_trajectory-buffer-randomized_2023-04-23_12-27-36",
        # "data/experiments/toy1d-cont_buffers/toy1d_cont_rand_multitask-buffer-randomized_2023-04-23_12-29-24",

        # "/home/ubuntu/juan/Meta-RL/data/toy1d_rand_Base-config_2023-11-14_10-41-02", 
        # "/home/ubuntu/juan/Meta-RL/data/randomization_experiment_2023-11-14_18-22-52",
        # "/home/ubuntu/juan/Meta-RL/data/toy1d_rand_Base-config_2023-11-14_12-27-45",
        # "/home/ubuntu/juan/Meta-RL/data/randomization_experiment_mult:0.5-2"
        # "/home/ubuntu/juan/Meta-RL/data/randomization_experiment_mult:0.3-5",
        # "/home/ubuntu/juan/Meta-RL/data/randomization_experiment_mult:0.5-2"
        # "/home/ubuntu/juan/Meta-RL/data/Huge_experiment_2023-11-14_17-29-32"
        # "/home/ubuntu/juan/Meta-RL/data/Huge_experiment-random-0.5-2_2023-11-15_19-08-11"
        # "/home/ubuntu/juan/Meta-RL/data/randomization_experiment_mult:0.8-1.5",
        # "/home/ubuntu/juan/Meta-RL/data/randomization_experiment_mult:0.2-10"
        # "/home/ubuntu/juan/Meta-RL/data/Huge_experiment_mult:0.3-5_2023-11-16_11-44-36"
        # "/home/ubuntu/juan/Meta-RL/data/randomization_experiment_mult:0.9-1.1",
        # "/home/ubuntu/juan/Meta-RL/data/randomization_experiment_mult:0.9-1.1(toy1d_rand)",
        # "/home/ubuntu/juan/Meta-RL/data/Hardcoded 0.5-1.5_2023-11-18_11-17-48"
        # "/home/ubuntu/juan/Meta-RL/data/Hardcoded 0.5-1.5_2023-11-16_12-58-32"
        # "/home/ubuntu/juan/Meta-RL/data/fails/Hardcoded 0.5-1.5 symmetric_2023-11-16_13-00-08"
        # "/home/ubuntu/juan/Meta-RL/data/Hardcoded 0.5-1.5 symmetric_2023-11-18_13-16-14"
        # "/home/ubuntu/juan/Meta-RL/data/Hardcoded 0.5-1.5 symmetric_2023-11-18_13-16-14_2023-11-18_19-39-18_transfer"
        # "/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43"
        # "/home/ubuntu/juan/Meta-RL/data/Baseline with multiplier_2023-11-18_19-59-28",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_2023-11-18_19-42-43"
        # "/home/ubuntu/juan/Meta-RL/data/Baseline with multiplier_2023-11-18_19-59-28_transfer",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_2023-11-18_19-42-43_transfer"
        # "/home/ubuntu/juan/Meta-RL/data/Multiplier with multi memory(exploration)_2023-11-19_11-47-48",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline multi memory(exploration)_2023-11-19_11-46-39"
        # "/home/ubuntu/juan/Meta-RL/data/Multiplier with multi memory(exploration)_2023-11-19_
        # "/home/ubuntu/juan/Meta-RL/data/cheetah_goal_TE-log_2023-11-28_21-48-27"
        # "/home/ubuntu/juan/Meta-RL/data/transfer_huge_small_steps/toy1d_rand_huge_TE-toy1d_2023-12-01_16-21-44"
        # "/home/ubuntu/juan/Meta-RL/data/transfer_huge_small_steps/test/toy1d_rand_huge_TE-toy1d_2023-12-01_16-30-31"
        # "/home/ubuntu/juan/Meta-RL/data/transfer_huge_small_steps/toy1d_rand_smaller_TE-toy1d_2023-12-01_16-36-10"
        # "/home/ubuntu/juan/Meta-RL/data/transfer_huge_small_steps/toy1d_TE-toy1d_2023-12-02_15-18-57"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/cheetah_goal_TE-log-strided_2023-12-01_16-06-37"
        # "/home/ubuntu/juan/Meta-RL/data/transfer_huge_small_steps/toy1d_TE-toy1d_2023-12-02_16-25-19"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/cheetah_goal_TE-log-strided_2023-12-07_12-56-31"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/cheetah_goal_TE-log-strided_2023-12-07_13-27-58"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/cheetah_goal_TE-log-strided_2023-12-07_13-57-02"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/averesto/cheetah_goal_TE-log-strided_2023-12-07_14-05-38"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/test/toy1d_TE-log_2023-12-07_19-01-10"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/cheetah_goal_TE-log-strided_2023-12-07_13-35-04"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/test/toy1d_TE-log_2023-12-07_19-09-31"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/test/toy1d_TE-log_2023-12-07_19-14-03"
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_50_small_steps_2023-12-07_19-07-13",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_50_2023-12-07_18-40-02"
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_50_big_steps_2023-12-08_09-26-47"
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_1_steps_0.01_2023-12-08_10-09-58"
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_1_steps_0.01_2023-12-08_10-09-58",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_50_2023-12-07_18-40-02",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_50_big_steps_2023-12-08_09-26-47",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_50_small_steps_2023-12-07_19-07-13"
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_unifrom_sampling_-10_10_2023-12-08_17-19-23",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_increased_S_3_2023-12-09_11-32-46",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_increased_S_5_2023-12-09_11-31-52",
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/test/cheetah_goal_TE-log-strided_2023-12-10_11-01-32"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/test/cheetah_goal_TE-log-strided_2023-12-10_11-24-59"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/test/cheetah_goal_TE-log_2023-12-10_11-50-37"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/test/cheetah_goal_TE-log-strided_2023-12-10_13-46-13"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/test/cheetah_goal_TE-log-strided_2023-12-10_13-53-38",
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/test/cheetah_goal_TE-log_2023-12-10_13-56-57"
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_50_2023-12-07_18-40-02"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/test/cheetah_goal_TE-log_2023-12-10_14-21-43"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/test/cheetah_goal_TE-log-strided_2023-12-10_14-26-20"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/test/toy1d_TE-log_2023-12-10_17-09-10"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/test/toy1d_TE-log_2023-12-10_17-15-20"
        # "/home/ubuntu/juan/base/data/Baseline_exloration"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/no_pretrained/cheetah_goal_TE-log-strided(500-100)_2023-12-10_20-47-20",
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/pretrained/cheetah_goal_TE-log-strided(500-100)_2023-12-10_21-14-53",
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/pretrained/cheetah_goal_TE-log-strided(500-100)_std_5_2023-12-11_19-24-44",
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/full/np_pretrained/cheetah_goal_TE-log-strided(500-100)_std_5_2023-12-11_19-22-08"
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_50_2023-12-07_18-40-02",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_increased_S_3_2023-12-09_11-32-46",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_increased_S_5_2023-12-09_11-31-52",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_1_steps_0.01_2023-12-08_10-09-58"
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_increased_policy_expl_var_S_3_2023-12-13_11-29-31"
        # "/home/ubuntu/juan/Meta-RL/data/delete/test_2023-12-18_15-31-19",
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/test/toy1d_TE-log_2023-12-18_16-32-08"
        # "/home/ubuntu/juan/Meta-RL/data/delete/test_2023-12-18_16-51-42",
        # "/home/ubuntu/juan/Meta-RL/data/delete/test_2023-12-18_17-50-31",
        # "/home/ubuntu/juan/Meta-RL/data/presentation/baseline_2023-12-20_10-28-02",
        # "/home/ubuntu/juan/Meta-RL/data/presentation/baseline_task50_action1_2023-12-20_10-29-20",
        # "/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_50_2023-12-07_18-40-02"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/test/toy1d_TE-log_2023-12-20_12-27-09"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/presentation/toy1d_TE-log_2023-12-20_16-28-57"
        # "/home/ubuntu/juan/Meta-RL/data/transfer/cheetah/presentation/cheetah_goal_TE-log-strided(500-100)_uniform_2023-12-20_16-35-11"
        # '/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/task10_action3_2024-01-19_11-46-08',
        # '/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/task10_action3_exploration_2024-01-20_00-02-46'
        # '/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/task10_action3_little_exploration_2024-01-20_09-23-28',
        # '/home/ubuntu/juan/data/pretrained_for_transfer/baseline_2024-01-20_15-33-32/'
        # '/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/task10_actoin3_2024-01-21_12-49-25',
        # '/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/task10_actoin3_toy1d_rand_2024-01-21_12-49-08'
        # '/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/higher_exploration(S_3)_action1/_2024-01-22_16-36-08',
        # '/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/higher_exploration(S_3)_new_policy/_2024-01-22_16-10-02',       # new exploration policy
        # '/home/ubuntu/juan/data/pretrained_for_transfer/baselinexploration_exploration_action1/_2024-01-23_07-47-08/',          # action 1
        # '/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/simple_exploration_action1/_2024-01-23_11-04-04',               # simple_exploration (ranom_policy)
        # '/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/toy1d_TE-toy1d_2024-01-28_10-20-52',                            # simple_exploration continue training
        # '/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/exploration_only_policy/_2024-01-28_11-25-10',                    # Exploration only in policy
        # '/home/ubuntu/juan/data/pretrained_for_transfer/exploration_only_policy/50_policy_loops/_2024-01-28_12-30-29/',           # Increase policy steps
        # '/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/exploration_only_policy/one_direction/_2024-01-29_16-28-52',      # One sided task
        # # '/home/ubuntu/juan/data/pretrained_for_transfer/exploration_only_policy/one_direction/_2024-01-31_08-29-43/',             # One sided, faster
        # '/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/toy1d_TE-toy1d_2024-02-04_11-27-23',                              # one sided faster, trained on more epochs
        # '/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/toy1d_TE-toy1d_2024-02-04_11-32-43',                              # Only train policy
        # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/simple_exploration/_2024-04-23_11-17-56',    # simple_exploration
        # # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/simple_no_exploration/_2024-04-23_11-14-14', # simple no_exploration
        # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/simple_exploration_only_positive/_2024-04-23_18-57-09',   # simple exploration, only positive 0.5
        # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/simple_exploration_only_positive_1/_2024-04-24_08-55-08',    # increase step size to 1
        # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/simple_exploration_biggerNN/_2024-04-24_11-13-30',   # bigger NN, action 0.5
        # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/simple_exploration_biggerNN_-1_1/_2024-04-24_11-14-41',  # bigger NN, action 0.1, state space -1,1
        # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/step1_biggerNN_-10_10/_2024-04-25_10-03-11',     # step 1, -10,10, NN[64,64,64,64]
        # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/step1_NN[300]_-10_10/_2024-04-25_10-02-48',      # step 1, -10,10, NN[300,300,300]
        # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/step1_biggerNN_-10_10/_2024-05-04_11-03-56',     # step 0.01, time steps 2500
        # # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/step1_biggerNN_-10_10/_2024-05-04_11-04-50',     # step 0.05, time steps 2500
        # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/step1_biggerNN_-10_10/_2024-05-04_17-09-16',     # no change task
        # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/step1_biggerNN_-10_10/_2024-05-13_16-31-32',     # same as /home/ubuntu/juan/Meta-RL/data/experiments_thesis/simple_exploration_biggerNN/_2024-04-24_11-13-30 but with no exploration
        # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/step1_biggerNN_-10_10/_2024-05-14_21-24-16',     # same as above, above not good
        # '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/step1_biggerNN_-10_10/_2024-05-15_09-16-38',     # remove also alpha
        '/home/ubuntu/juan/Meta-RL/data/experiments_thesis/step1_biggerNN_-10_10/_2024-05-16_10-19-16',     # remove alpha


        # "data/transfer_encoders/toy1d_attention-np",
        # "data/transfer_encoders/toy1d_log-rand",

    ]

    for path in paths:
        model_evaluation(
            path,
            save_dir='./evaluation/experiments_thesis/no_exploration/2.0',
            # save_dir = './data/delete',
            create_video=False,
            figure_size=(8,6),
            trajectory_2d=False,
            color_by=None,
        )