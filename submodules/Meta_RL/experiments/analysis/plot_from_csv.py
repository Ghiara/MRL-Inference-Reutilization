"""
This module helps to recreate plots which were created by 
'model_evaluation.py > model_evaluation()'. It defines functions which 
take the raw data of the figures, stored as CSV files, to plot the training
progress, trajectories, and latent space. 

The CSV-data which should be used is automatically created by ``model_evaluation()``
and stored alongside the evaluation images as
- "progress.csv": Training progress data
- Subdirectory "trajectories": Contains data of multiple trajectories.

Note that the plots are not necessarily identical in appearance to the original
figures! The functions below serve as an entry point to load and visualize data
the data from previous evaluations. They are meant to be adapted to your 
visual preferences. 
Additionally, this module aims to be as independent from additional submodules
as possible. You might want to use the "MRL-analysis-tools" package for plotting
utility. Ask me (Julius Durmann) if you need access.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-05-05
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path


figure_size = (20, 10)

# Colors
c_avg = 'steelblue'
c_max = 'chartreuse'
c_min = 'tomato'

"""
Utility functions (mainly redefine functions from MRL-analysis-tools)
"""

def smooth_data(y: np.ndarray, scale_param: float = 5.0, kernel_width: int = 15) -> np.ndarray:
    """Compute a smoothed version of the input data by applying an exponential
    kernel.

    Parameters
    ----------
    y : np.ndarray
        Data
    scale_param : float, optional
        Scale parameter of the kernel.
        Large values lead to stronger smoothing, by default 5.0
    kernel_width : int, optional
        Maximum width (hard limit) of the kernel, by default 15

    Returns
    -------
    np.ndarray
        Smoothed data
    """
    
    # Input parsing
    kernel_width = int(kernel_width)
    if kernel_width % 2 == 0:
        # Make sure that kernel window is centered around middle -> odd number!
        kernel_width += 1
    y = np.array(y)

    # Exponential kernel
    kernel = np.arange(-kernel_width//2+1, kernel_width//2+1)
    kernel = np.exp(-np.square(kernel/scale_param))
    kernel /= kernel.sum()  # Normalization

    # Pad boundaries to avoid boundary effects
    # y = np.concatenate([y[kernel_width//2:0:-1], y, y[-1:-kernel_width//2:-1]], axis=0)
    y = np.concatenate([y[0].repeat(kernel_width//2), y, y[-1].repeat(kernel_width//2)], axis=0)

    # Apply kernel for smoothing
    return np.convolve(kernel, y, 'valid')

def smooth_plot(axs: plt.Axes, x: np.ndarray, y: np.ndarray, kernel_width: int = None, scale_param: int = None, label: str = None, plot_original: bool = True, **kwargs):
    """Plot smoothed (x,y)-data.

    Parameters
    ----------
    axs : plt.Axes
        Axis which is used to plot the data.
    x : np.ndarray
        X-values
    y : np.ndarray
        Y-values (will be smoothed).
    kernel_width : int, optional
        Width of the kernel, see ``smooth_data()``, by default None
    scale_param : int, optional
        Scale parameter of the kernel, see ``smooth_data()``, by default None
    label : str, optional
        Label of the provided data, by default None
    plot_original : bool, optional
        Set to False to avoid plotting the original values, by default True
    **kwargs
        Additional parameters for ``axs.plot()``
    """
    if kernel_width is None: kernel_width = int(len(x) / 10)
    if scale_param is None: scale_param = len(x) / 50
    y_ = smooth_data(y, scale_param, kernel_width)
    line = axs.plot(x, y_, label=label, **kwargs)[0]
    if plot_original:
        try:
            a = kwargs['alpha']
            del kwargs['alpha']
        except KeyError:
            a = 1.0
        kwargs['color'] = line.get_color()
        axs.plot(x, y, alpha=0.3 * a, **kwargs)

def smooth_fill_between(axs: plt.Axes, x: np.ndarray, min_y: np.ndarray, max_y: np.ndarray, kernel_width: int = None, scale_param: int = None, alpha: float = 0.3, **kwargs):
    """Smoothed version of ``axs.fill_between()``. Uses ``smooth_data()`` to 
    create smoothed y-values.

    Parameters
    ----------
    axs : plt.Axes
        Axis which is used to plot the data.
    x : np.ndarray
        X-values
    min_y : np.ndarray
        Y-values for lower boundary
    max_y : np.ndarray
        Y-values for upper boundary
    kernel_width : int, optional
         Width of the kernel, see ``smooth_data()``, by default None
    scale_param : int, optional
        Scale parameter of the kernel, see ``smooth_data()``, by default None
    alpha : float, optional
        Transparancy value of the surface. See matplotlib library, by default 0.3
    **kwargs
        Additional parameters for ``axs.fill_between()``
    """
    if kernel_width is None: kernel_width = int(len(x) / 10)
    if scale_param is None: scale_param = len(x) / 50
    min_y = smooth_data(min_y, scale_param, kernel_width)
    max_y = smooth_data(max_y, scale_param, kernel_width)
    axs.fill_between(x, min_y, max_y, alpha=alpha, **kwargs)


def show_save_figure(fig: Figure = None, save_path: str = None, show: bool = False):
    print(save_path)
    if save_path is not None:
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)
    print(show)
    if show:
        plt.show()



"""
Plotting functions
"""

def plot_training_curves(file: str):
    """Plot training progress

    Parameters
    ----------
    file : str
        Filename with training progress data (usually "progress.csv")
    """
    # Load data
    progress: pd.DataFrame = pd.read_csv(file)

    # Policy training
    try:
        print("Policy training plots ...")

        x = progress['trainer/Policy trainer/num train calls']

        return_fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_fill_between(axs, x, progress['eval/Returns Min'], progress['eval/Returns Max'], color='lightgrey', alpha=0.3)
        smooth_plot(axs, x, progress['eval/Average Returns'], color=c_avg, label='Average returns')
        smooth_plot(axs, x, progress['eval/Returns Max'], color=c_max, label='Maximum returns', linestyle='dashed', plot_original=False)
        smooth_plot(axs, x, progress['eval/Returns Min'], color=c_min, label='Minimum returns', linestyle='dashed', plot_original=False)
        axs.set_xlabel('Train steps (policy)')
        axs.legend()

        action_figure, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Policy trainer/policy/normal/std Mean'], label="Standard deviation (mean)")
        smooth_plot(axs, x, progress['trainer/Policy trainer/policy/normal/std Min'], label="Standard deviation (min)")
        smooth_plot(axs, x, progress['trainer/Policy trainer/policy/normal/std Max'], label="Standard deviation (max)")
        axs.set_xlabel('Train steps (policy)')
        axs.legend()

        q_figure, axs = plt.subplots(1, figsize=figure_size)
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

        q_loss_figure, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Policy trainer/QF1 Loss'], label="QF1 loss")
        smooth_plot(axs, x, progress['trainer/Policy trainer/QF2 Loss'], label="QF2 loss")
        axs.set_xlabel('Train steps (policy)')
        axs.legend()
        axs.set_yscale('log')

        policy_loss_figure, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Policy trainer/Policy Loss'], label="Policy loss")
        axs.set_xlabel('Train steps (policy)')
        axs.legend()
    except KeyError:
        print("(skipped)")


    # Inference training
    try:
        print("Inference training plots ...")
        x = progress['trainer/Inference trainer/num train calls']

        elbo_figure, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Inference trainer/elbo'])
        axs.set_xlabel('Train steps (inference)')
        axs.legend(['ELBO'])

        prediction_error_figure, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Inference trainer/reward prediction error'], label="Reward prediction error")
        smooth_plot(axs, x, progress['trainer/Inference trainer/state prediction error'], label="State prediction error")
        axs.set_xlabel('Train steps (inference)')
        axs.set_yscale('log')
        axs.legend()
    except KeyError:
        print("(skipped)")

    show_save_figure(show=True)


def plot_trajectory(file: str, dim: int = 0, axs: plt.Axes = None, color: Any = None) -> plt.Axes:
    """Plot a single trajectory

    Parameters
    ----------
    file : str
        Filename with trajectory data (usually in subdirectory "trajectories")
    dim : int, optional
        Dimension of the trajectory which is plotted, by default 0
    axs : plt.Axes, optional
        Provide an axis handle to use for plotting. By default None
    """
    data = pd.read_csv(file, sep=",")
    traj = data[f'observations.{dim}']
    goal = data[f'task/goal.0']
    if axs is None:
        fig, axs = plt.subplots()
    axs.plot(np.arange(len(traj)), traj, c=color)
    axs.plot(np.arange(len(traj)), goal, c=color, linestyle='dashed')

    return axs


def plot_latent_space(trajectory_files: List[str]) -> Tuple[Figure, plt.Axes]:
    """Plot the latent space as a scatter plot, colored by the true goals.
    (Assumes goal-defined tasks.)

    For a more advanced plotting function, you might want to refer to 
    "plot_from_csv.py" in submodule 'MRL-analysis-tools > plots'. (This requires
    to install the submodule!)

    Parameters
    ----------
    trajectory_files : List[str]
        List of trajectory file names from which the latent space can be extracted.
        These files can be usually found in subdirectory "trajectories".
    """
    encodings_0 = []
    encodings_1 = []
    goals = []

    # Load the encoding data from the trajectory files
    for trajectory_file in trajectory_files:
        data = pd.read_csv(trajectory_file, sep=",")
        enc_0 = data['encodings.0']
        try:
            enc_1 = data['encodings.1']
        except:
            enc_1 = np.zeros_like(enc_0)
        goal = data[f'task/goal.0']
        encodings_0.append(enc_0)
        encodings_1.append(enc_1)
        goals.append(goal)

    # Append the encoding data to one-dimensional vectors
    encodings_0 = np.hstack(encodings_0)
    encodings_1 = np.hstack(encodings_1)
    goals = np.hstack(goals)

    # Scatter plot
    fig, axs = plt.subplots()
    axs.scatter(encodings_0, encodings_1, c=goals)

    return fig, axs

def plot_latent_over_goal(trajectory_files: List[str]) -> Tuple[Figure, plt.Axes]:
    """Plot the latent means over the true goal (= task).

    Parameters
    ----------
    trajectory_files : List[str]
        List of trajectory file names from which the latent space can be extracted.
        These files can be usually found in subdirectory "trajectories".
    """
    latents = []
    goals = []

    # Load the latent means and goals from trajectory files
    for trajectory_file in trajectory_files:
        data = pd.read_csv(trajectory_file, sep=",")
        latent = data['latent_mean.0']
        goal = data[f'task/goal.0']
        latents.append(latent)
        goals.append(goal)

    # Append the data to one-dimensional vectors
    latents = np.hstack(latents)
    goals = np.hstack(goals)

    # Scatter plot
    fig, axs = plt.subplots()
    axs.scatter(goals, latents)

    return fig, axs



if __name__ == "__main__":
    from pathlib import Path

    p = Path("{os.getcwd()}/evaluation/toy1d-cont_buffers/toy1d_rand_Base-config_2023-11-14_10-41-02/")
    
    print("Plotting training curves ...")
    plot_training_curves(p.joinpath("progress.csv"))

    print("Plotting trajectories ...")
    fig, axs = plt.subplots()
    for c, i in enumerate(range(0, 50, 5)):
        plot_trajectory(p.joinpath("trajectories").joinpath(f"{i+1}.csv"), dim=0, axs=axs, color=f"C{c}")
    show_save_figure(fig, show=True)

    print("Plotting latent space ...")
    fig, axs = plot_latent_space([f"/home/ubuntu/juan/Meta-RL/evaluation/toy1d-cont_buffers/toy1d_rand_Base-config_2023-11-14_10-41-02/trajectories/{i+1}.csv" for i in range(50)])
    show_save_figure(fig, show=True)

    fig, axs = plot_latent_over_goal([f"/home/ubuntu/juan/Meta-RL/evaluation/toy1d-cont_buffers/toy1d_rand_Base-config_2023-11-14_10-41-02/trajectories/{i+1}.csv" for i in range(50)])
    show_save_figure(fig, show=True)