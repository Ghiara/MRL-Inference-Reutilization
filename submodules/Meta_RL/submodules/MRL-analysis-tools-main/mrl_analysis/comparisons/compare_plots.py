from typing import List, Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

from mrl_analysis.utility.data_smoothing import smooth_fill_between, smooth_plot


def compare_progress(
    progress_data: List[pd.DataFrame],
    x_key: str, y_key: str,
    names: List[str] = None,
    std_dev_key: str = None,
    min_key: str = None,
    max_key: str = None,
    plot_kwargs: List[Dict] = None,
    plot_original: bool = True,
    xlim: Tuple[float,float] = None,
) -> Tuple[Figure, plt.Axes]:
    """Plot (multiple) training progress data, e.g. returns over epochs.

    Parameters
    ----------
    progress_data : List[pd.DataFrame]
        Set of training progress data
    x_key : str
        Key for accessing x-axis data in the DataFrames.
    y_key : str
        Key for accessing y-axis data in the DataFrames.
    names : List[str], optional
        Legend names, by default None
    std_dev_key : str, optional
        Key for accessing standard deviation data in DataFrames,
        by default None
    min_key : str, optional
        Key for accessing minimum values data in DataFrames,
        by default None
    max_key : str, optional
        Key for accessing maximum values data in DataFrames,
        by default None
    smooth : bool, optional
        Smooth plots by temporal convolution, by default False
    plot_kwargs : List[Dict], optional
        Arguments for the plots, only applied to lines (not surfaces)
    plot_original : bool, optional
        Set to False to only plot smoothed lines. By default True
    xlim : Tuple[int, int], optional
        Reduce data to be within provided boundaries

    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure handle of the plot
        Axes handle of the plot
    """
    if names is None:
        names = [f"Dataset {i+1}" for i in range(len(progress_data))]

    if plot_kwargs is None:
        plot_kwargs = [{} for i in range(len(progress_data))]
    for i, p_kwargs in enumerate(plot_kwargs):
        if not "color" in p_kwargs.keys():
            p_kwargs['color'] = f"C{i}"

    fig, axs = plt.subplots(1)

    for name, data, p_kwargs in zip(names, progress_data, plot_kwargs):
        if xlim is not None:
            data = data[data[x_key] >= xlim[0]]
            data = data[data[x_key] <= xlim[1]]
        x = data[x_key]
        y = data[y_key]
        
        if std_dev_key is not None:
            upper = data[y_key] + data[std_dev_key]
            lower = data[y_key] - data[std_dev_key]
            smooth_fill_between(axs, x, lower, upper, color=p_kwargs['color'], alpha=0.25, zorder=1)
        
        if min_key is not None and max_key is not None:
            upper = data[max_key]
            lower = data[min_key]
            smooth_fill_between(axs, x, lower, upper, color=p_kwargs['color'], alpha=0.25, zorder=2)

        if plot_original:
            axs.plot(x,y,color=p_kwargs['color'], alpha=0.3, zorder=3)

        smooth_plot(axs, x, y, label=name, plot_original=False, **p_kwargs, zorder=4)

    # if std_dev_key is not None:
    #     for name, data, p_kwargs in zip(names, progress_data, plot_kwargs):
    #         x = data[x_key]
    #         y = data[y_key]
    #         upper = data[y_key] + data[std_dev_key]
    #         lower = data[y_key] - data[std_dev_key]
    #         smooth_fill_between(axs, x, lower, upper, color=p_kwargs['color'], alpha=0.25)
    # if min_key is not None and max_key is not None:
    #     for name, data, p_kwargs in zip(names, progress_data, plot_kwargs):
    #         x = data[x_key]
    #         y = data[y_key]
    #         upper = data[max_key]
    #         lower = data[min_key]
    #         smooth_fill_between(axs, x, lower, upper, color=p_kwargs['color'], alpha=0.25)
    # if plot_original:
    #     for name, data, p_kwargs in zip(names, progress_data, plot_kwargs):
    #         x = data[x_key]
    #         y = data[y_key]
    #         axs.plot(x,y,color=p_kwargs['color'], alpha=0.3)
    # for name, data, p_kwargs in zip(names, progress_data, plot_kwargs):
    #     x = data[x_key]
    #     y = data[y_key]
    #     smooth_plot(axs, x, y, label=name, plot_original=False, **p_kwargs)

    axs.legend()
    axs.set_xlabel(x_key)
    axs.set_ylabel(y_key)

    return fig, axs
