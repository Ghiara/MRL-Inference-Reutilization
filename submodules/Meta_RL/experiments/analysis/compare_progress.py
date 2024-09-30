"""
This file contains the function ``compare()`` which visualizes training 
progress data, including returns, policy losses, and inference losses, of
multiple agents.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-04-06
"""

from typing import List, Dict, Tuple
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

from mrl_analysis.comparisons.compare_plots import compare_progress
from smrl.experiment.analysis import load_results
import matplotlib.pyplot as plt

from mrl_analysis.plots.plot_settings import *


def compare(
    paths: List[str], 
    savepath: str, 
    names: List[str] = None, 
    xlim: Tuple[float,float] = None, 
    figsize = (16,8), 
    plot_kwargs: List[Dict] = None,
    plot_original: bool = True,
):
    """Compare training curves of several training runs.

    Parameters
    ----------
    paths : List[str]
        Paths to the training data (directories!),
        See also ``smrl.experiment.analysis.load_results()``.
    savepath : str
        Path where the plots should be saved
    names : List[str], optional
        Name of the training runs, by default None
    xlim : Tuple[float, float], optional
        Provide limits to cut the data, by default None
    figsize : tuple, optional
        Size of the figure. Important for image resolution, by default (16,8)
    plot_kwargs : List[Dict], optional
        Additional arguments for the plots (``matplotlib.pyplot.plot()``), 
        by default None
    plot_original : bool, optional
        Set to False to only plot smoothed curves, by default True
    """

    savepath = Path(savepath)

    results = [load_results(p) for p in paths]
    configs = [result['config'] for result in results]
    if names is None:
        names = [config['description']['name'] for config in configs]
    progress_data = [result['progress'] for result in results]

    # RETURNS
    print("Return plot")
    try:
        min_key, max_key, std_dev_key = None, None, None

        x_key = "trainer/Policy trainer/num train calls"
        y_key = "eval/Returns Mean"
        # std_dev_key = "eval/Returns Std"
        # min_key = "eval/Returns Min"
        # max_key = "eval/Returns Max"

        fig, axs = compare_progress(
            progress_data=progress_data, 
            x_key=x_key, 
            y_key=y_key,
            names=names, 
            std_dev_key=std_dev_key, 
            min_key=min_key, 
            max_key=max_key,
            plot_kwargs=plot_kwargs,
            plot_original=plot_original,
            xlim=xlim,
        )
        # if xlim is not None:
        #     axs.set_xlim(xlim)
        fig.set_size_inches(*figsize)
        fig.tight_layout()

        axs.set_ylabel("Returns (evaluation)")
        axs.set_xlabel("Train steps (policy)")

        fig.savefig(savepath.joinpath("comparison_returns.png"))
        fig.savefig(savepath.joinpath("comparison_returns.svg"))
        fig.savefig(savepath.joinpath("comparison_returns.pdf"))
        print("Done")
    except KeyError:
        print("Skipped")


    # POLICY LOSS
    try:
        print("Policy loss plot")
        x_key = "trainer/Policy trainer/num train calls"
        y_key = "trainer/Policy trainer/Policy Loss"

        fig, axs = compare_progress(
            progress_data=progress_data, 
            x_key=x_key, 
            y_key=y_key,
            names=names, 
            std_dev_key=std_dev_key, 
            min_key=min_key, 
            max_key=max_key,
            plot_kwargs=plot_kwargs,
            plot_original=plot_original,
            xlim=xlim,
        )
        # if xlim is not None:
        #     axs.set_xlim(xlim)
        fig.set_size_inches(*figsize)
        fig.tight_layout()

        axs.set_ylabel("Policy loss")
        axs.set_xlabel("Train steps (policy)")

        fig.savefig(savepath.joinpath("comparison_policy-loss.png"))
        fig.savefig(savepath.joinpath("comparison_policy-loss.svg"))
        fig.savefig(savepath.joinpath("comparison_policy-loss.pdf"))
        print("Done!")
    except KeyError:
        print("Skipped")


    # REWARD PREDICTION ERROR
    try:
        print("Prediction error plot")
        x_key = "trainer/Inference trainer/num train calls"
        y_key = "trainer/Inference trainer/reward prediction error"

        fig, axs = compare_progress(
            progress_data=progress_data, 
            x_key=x_key, 
            y_key=y_key,
            names=names, 
            std_dev_key=std_dev_key, 
            min_key=min_key, 
            max_key=max_key,
            plot_kwargs=plot_kwargs,
            plot_original=plot_original,
            xlim=xlim,
        )
        # if xlim is not None:
        #     axs.set_xlim(xlim)
        fig.set_size_inches(*figsize)
        fig.tight_layout()
        axs.set_yscale('log')
        axs.set_ylabel("Prediction error (reward)")
        axs.set_xlabel("Train steps (inference)")

        fig.savefig(savepath.joinpath("comparison_prediction-error.png"))
        fig.savefig(savepath.joinpath("comparison_prediction-error.svg"))
        fig.savefig(savepath.joinpath("comparison_prediction-error.pdf"))
        print("Done")
    except KeyError:
        print("Skipped")


    # x_key = "trainer/Policy trainer/num train calls"
    # y_key = "trainer/Policy trainer/QF1 Loss"
    # # y_key = "trainer/Policy trainer/Q1 Predictions Mean"



if __name__ == "__main__":
    # Load models
    paths = [

        # "data/experiments/toy2d_on-off-policy/toy2d_rand_on-policy_2023-02-26_19-56-47",
        # "data/experiments/toy2d_on-off-policy/toy2d_rand_off-policy_random-inference_2023-02-26_19-57-44",
        # "data/experiments/toy2d_on-off-policy/toy2d_rand_off-policy_memory-random-inference_2023-02-26_19-57-52",
        # "data/experiments/toy2d_on-off-policy/toy2d_rand_off-policy_multi-random-inference_2023-02-26_19-57-57",
        # "data/experiments/toy2d_on-off-policy/toy2d_rand_off-policy_log-random-inference_2023-02-26_19-58-01",
        # "data/experiments/toy2d_on-off-policy/toy2d_off-policy_log-random-inference_2023-02-26_19-58-14",
        # "data/experiments/toy2d_on-off-policy/toy2d_rand_on-policy-entropy-tuning_2023-02-26_20-05-57",

        # "data/experiments/toy1d_buffers/toy1d_rand_context-buffer_2023-03-05_10-55-36",
        # "data/experiments/toy1d_buffers/toy1d_rand_trajectory-buffer-ordered_2023-03-05_10-56-19",
        # "data/experiments/toy1d_buffers/toy1d_rand_trajectory-buffer-randomized_2023-03-05_10-57-00",
        # "data/experiments/toy1d_buffers/toy1d_rand_multitask-buffer-ordered_2023-03-05_10-57-52",
        # "data/experiments/toy1d_buffers/toy1d_rand_multitask-buffer-randomized_2023-03-05_10-58-27",

        # "data/experiments/toy1d_on-off-policy/toy1d_rand_on-policy_2023-03-15_20-21-08",
        # "data/experiments/toy1d_on-off-policy/toy1d_rand_off-policy_random-inference_2023-03-16_14-38-41",
        # # "data/experiments/toy1d_on-off-policy/toy1d_rand_off-policy_memory-random-inference_2023-03-17_20-26-11",
        # "data/experiments/toy1d_on-off-policy/toy1d_rand_off-policy_memory-random-inference_2023-03-19_09-52-33",
        # "data/experiments/toy1d_on-off-policy/toy1d_rand_off-policy_multi-random-inference_2023-03-16_14-11-19",
        # "data/experiments/toy1d_on-off-policy/toy1d_rand_off-policy_log-random-inference_2023-03-15_20-20-29",

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

        # "data/experiments/toy1d_encoders/toy1d_rand_mlp-encoder-vae_2023-03-21_09-52-45",
        # # "data/experiments/toy1d_encoders/toy1d_rand_pair-encoder-vae_2023-03-21_16-57-06",
        # # "data/experiments/toy1d_encoders/toy1d_rand_pair-encoder-np_2023-03-20_11-49-14",
        # "data/experiments/toy1d_encoders/toy1d_rand_gru-encoder-vae_2023-03-20_11-51-19",
        # "data/experiments/toy1d_encoders/toy1d_rand_gru-encoder-np_2023-03-22_05-25-36",
        # "data/experiments/toy1d_encoders/toy1d_rand_attention-encoder-vae_2023-03-24_09-38-00",
        # "data/experiments/toy1d_encoders/toy1d_rand_attention-encoder-np_2023-03-24_09-38-33",

        # "data/experiments/toy1d-cont_on-off-policy/toy1d_cont_rand_on-policy_2023-03-17_11-35-43",
        # # "data/experiments/toy1d-cont_on-off-policy/toy1d_cont_rand_on-policy_2023-03-24_10-48-59",
        # "data/experiments/toy1d-cont_on-off-policy/toy1d_cont_rand_off-policy_random-inference_2023-03-20_10-42-23",
        # # "data/experiments/toy1d-cont_on-off-policy/toy1d_cont_rand_off-policy_random-inference_2023-03-25_20-21-35",
        # "data/experiments/toy1d-cont_on-off-policy/toy1d_cont_rand_off-policy_memory-random-inference_2023-03-21_18-39-59",
        # # "data/experiments/toy1d-cont_on-off-policy/toy1d_cont_rand_off-policy_memory-random-inference_2023-03-27_05-33-40",
        # "data/experiments/toy1d-cont_on-off-policy/toy1d_cont_rand_off-policy_multi-random-inference_2023-03-20_10-43-07",
        # # "data/experiments/toy1d-cont_on-off-policy/toy1d_cont_rand_off-policy_multi-random-inference_2023-03-25_11-28-42",
        # "data/experiments/toy1d-cont_on-off-policy/toy1d_cont_rand_off-policy_log-random-inference_2023-03-21_17-27-54",
        # # "data/experiments/toy1d-cont_on-off-policy/toy1d_cont_rand_off-policy_log-random-inference_2023-03-26_15-55-57",

        # "data/experiments/toy1d-cont_on-off-policy_3/toy1d_cont_rand_on-policy_2023-03-28_11-49-49",
        # "data/experiments/toy1d-cont_on-off-policy_3/toy1d_cont_rand_off-policy_random-inference_2023-03-29_09-44-05",
        # "data/experiments/toy1d-cont_on-off-policy_3/toy1d_cont_rand_off-policy_memory-random-inference_2023-03-28_11-50-37",
        # "data/experiments/toy1d-cont_on-off-policy_3/toy1d_cont_rand_off-policy_multi-random-inference_2023-03-28_10-17-55",
        # "data/experiments/toy1d-cont_on-off-policy_3/toy1d_cont_rand_off-policy_log-random-inference_2023-03-28_10-20-35",

        # "data/experiments/toy1d-cont_buffers/toy1d_cont_rand_trajectory-buffer-ordered_2023-04-23_10-53-39",
        # "data/experiments/toy1d-cont_buffers/toy1d_cont_rand_trajectory-buffer-randomized_2023-04-23_12-27-36",
        # "data/experiments/toy1d-cont_buffers/toy1d_cont_rand_multitask-buffer-randomized_2023-04-23_12-29-24",

        "/home/ubuntu/juan/Meta-RL/data/randomization_experiment_2023-11-14_18-22-52",
        "/home/ubuntu/juan/Meta-RL/data/toy1d_rand_Base-config_2023-11-14_12-27-45"

        # "data/experiments/toy1d-cont_buffers/toy1d_cont_rand_trajectory-buffer-randomized_2023-04-23_12-27-36",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-log-strided_2023-03-20_16-47-25",
        # "data/experiments/toy1d_TE_disc-to-cont/toy1d_cont_rand_TE-toy1d-attention-strided_2023-03-29_13-07-04",

    ]

    names = [
        # "On policy inference training",
        # # "On policy inference training",
        # "Off-policy inference training, no memory",
        # # "Off-policy inference training, no memory",
        # "Off-policy inference training, memory",
        # # "Off-policy inference training, memory",
        # "Off-policy inference training, multi-memory",
        # # "Off-policy inference training, multi-memory",
        # "Off-policy inference training, log-memory",
        # # "Off-policy inference training, log-memory",

        # "Context buffer (subsequent)",
        "Single trajectory (subsequent)",
        "Single trajectory (randomized)",
        # "Multiple trajectories (subsequent)",
        "Multiple trajectories (randomized)",

        # "On-policy",
        # "On-policy & striding",
        # "Random policy",
        # "Random policy & striding",
        # "Memory policy",
        # "Memory policy & striding", 
        # "Multiple random policies",
        # "Multiple random policies & striding",
        # "Log-scaled policies",
        # "Log-scaled policies & striding",

        # "MLP encoder + VAE",
        # # "Pair encoder + VAE",
        # # "Pair encoder + NP",
        # "GRU encoder + VAE",
        # "GRU encoder + NP",
        # "Attention encoder + VAE",
        # "Attention encoder + NP",

        # "Regularly trained",
        # "Encoder transfer",
        # "Encoder transfer (attention)",


    ]

    plot_kwargs = [
        {'color': "C0"},
        # {'linestyle': 'dashed', 'color': "C0"},
        {'color': "C1"},
        # {'linestyle': 'dashed', 'color': "C1"},
        {'color': "C2"},
        # {'linestyle': 'dashed', 'color': "C2"},
        {'color': "C3"},
        # {'linestyle': 'dashed', 'color': "C3"},
        {'color': "C4"},
        # {'linestyle': 'dashed', 'color': "C4"},

        # {'color': "C5"},
        # {'linestyle': 'dashed', 'color': "C5"},
        # {'color': "C6"},
        # {'linestyle': 'dashed', 'color': "C6"},
        # {'color': "C7"},
        # {'linestyle': 'dashed', 'color': "C7"},
        # {'color': "C8"},
        # {'linestyle': 'dashed', 'color': "C8"},
        # {'color': "C9"},
        # {'linestyle': 'dashed', 'color': "C9"},
    ]
    compare(
        paths, 
        names=names, 
        savepath="evaluation/toy1d-cont_buffers", 
        plot_kwargs=plot_kwargs, 
        plot_original=True,
        # xlim=[0, 50_000],
    )

