import numpy as np
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from sklearn.manifold import TSNE

MIN_LIGHTNESS = 0.1
MAX_LIGHTNESS = 0.9

def plot_latent_space(encodings: np.ndarray, tasks: np.ndarray, size=10.0) -> Tuple[Figure, List[plt.Axes]]:
    """Plot latent space encodings. Coloring is based on tasks indicators.

    Parameters
    ----------
    encodings : np.ndarray
        Latent space encodings, shape (n_samples, encoding_dim)
    tasks : List[float]
        Task specifications, used for coloring the samples
    size : float, optional 
        Size of the figure will be (size * task_dims, size), by default 10.0
        See matplotlib.pyplot.subplots' argument ``figsize``

    Returns
    -------
    Tuple[Figure, List[plt.Axes]] 
        Figure handle
        List of axes handle(s)
    """
    assert encodings.ndim == 2, "Input ``encodings`` must have two dimensions"

    # Ensure two-dimensional representation
    if encodings.shape[1] == 1:
        encodings = np.pad(encodings, ((0,0),(0,1)))
    if encodings.shape[1] > 2:
        encodings = TSNE().fit_transform(encodings)

    if tasks.ndim == 1:
        tasks = tasks[..., None]

    figsize = (tasks.shape[1]*size,size) if tasks.shape[1] > 1 else (1.5*size, size)
    
    fig, ax = plt.subplots(1, tasks.shape[1], figsize=figsize)
    ax = np.atleast_1d(np.array(ax))   # Ensure that indexing is possible
    for task_axis in range(tasks.shape[1]):
        scatters = ax[task_axis].scatter(encodings[:, 0], encodings[:, 1], c=tasks[:,task_axis], cmap=cm.gist_rainbow)
        divider = make_axes_locatable(ax[task_axis])
        # cax = divider.append_axes('bottom', size='5%', pad=0.05)
        cax = divider.append_axes('bottom', size='5%', pad="10%")
        cbar = fig.colorbar(scatters, cax=cax, extend='both', orientation='horizontal')
        # cbar = fig.colorbar(scatters, ax=ax[task_axis], location='bottom')
        if tasks.shape[1] > 1:
            ax[task_axis].set_title(f"Task dimension {task_axis+1}")
    cbar.minorticks_on()

    # fig.suptitle("Encoding space")

    return fig, ax


def plot_latent_space_axis_over_task(
    encodings: np.ndarray,
    tasks: Union[List[float], np.ndarray],
    color_information: Union[List[float], np.ndarray] = None,
) -> Tuple[Figure, plt.Axes]:
    """Plot one axis of the latent space encodings over one axis of the task specification. 
    Coloring is based on context variables.

    Parameters
    ----------
    encodings : np.ndarray
        Latent space encodings, used as y-axis, shape (n_samples, 1)
    tasks : Union[List[float], np.ndarray]
        Task specifications, used as x-axis, shape (n_samples, 1)
    color_information : Union[List[float], np.ndarray], optional
        Color values (float) for the scatter dots, shape (n_samples)
        
    Returns
    -------
    Tuple[plt.figure.Figure, plt.Axes] 
        Figure handle
        Axes handle
    """

    tasks = np.array(tasks)
    encodings = np.array(encodings)
        
    fig, ax = plt.subplots(1)
    scatters = ax.scatter(tasks[:,0], encodings[:,0], c=color_information, alpha=0.25)

    if color_information is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "3%", pad="3%")
        cbar = fig.colorbar(scatters, cax=cax, extend='both')
        cbar.minorticks_on()

    ax.set_title("Encoding vs. true task")
    ax.set_xlabel("True task")
    ax.set_ylabel("Encoding")

    return fig, ax