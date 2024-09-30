import numpy as np
import matplotlib.pyplot as plt

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
        Axes which is used to plot the data.
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