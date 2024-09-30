"""
This module contains the class ``InteractiveImagePlot``.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-25
"""

from typing import List, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


class InteractivePlot():
    """Class which allows interactive plots, i.e. plots which 
    can be updated.

    Parameters
    ----------
    figure : matplotlib.figure.Figure, optional
        Figure handle if existing plot should be used
    axis : plt.Axes, optional
        Axis handle if existing plot should be used
    zoom_mode : str, optional
        How y-axis limits are updated, can be one of 
        | ``'adaptive'`` | ``'keep'`` | ``'auto'`` |,
        by default 'adaptive'
    blit : bool, optional
        Set to True to signficantly speed up drawing, at the cost of no axis
        updates (i.e. ticks, etc. are not updated).
        By default False
    """
    def __init__(self, figure: Figure = None, axis: plt.Axes = None, zoom_mode: str = 'adaptive', blit: bool = False) -> None:
        if figure is not None and axis is not None:
            self.figure, self.axis = figure, axis
        else:
            self.figure, self.axis = plt.subplots(1)
        self._lines: List[plt.Line2D] = []
        self._surfaces: List = []
        self.zoom_mode = zoom_mode
        self.y_min, self.y_max = None, None
        self.eps = 0.1

        plt.show(block=False)
        figure.canvas.draw()
        figure.canvas.flush_events()
        self._background = self.figure.canvas.copy_from_bbox(self.axis.bbox)
        self.blit = blit

    def plot(self, x_data: np.ndarray, y_data: np.ndarray, labels: Union[str, List[str]] = None, *args, **kwargs):
        """Set displayed data.

        Parameters
        ----------
        image : np.ndarray
            Image array
        """
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        assert x_data.ndim == 1, "x_data must be a one-dimensional array"
        if y_data.ndim == 1:
            y_data = y_data[..., None]
        assert y_data.ndim == 2, "y_data can be either one or two dimensional!"
        
        if labels is None:
            labels = ["no label" for _ in range(y_data.shape[1])]
        if not isinstance(labels, list):
            labels = list(labels)

        for i, (y, label) in enumerate(zip(np.rollaxis(y_data, 1), labels)):
            if len(self._lines) <= i:
                line, = self.axis.plot(x_data, y, label=label, color=f"C{i}")
                self._lines.append(line)
            else:
                self._lines[i].set_data(x_data, y)
                self._lines[i].set_label(label)
            self.axis.legend()

        self._update()

    def _update(self):
        self.axis.relim()
        self.axis.autoscale()
        y_min, y_max = self.axis.get_ylim()
        if self.y_min is None or self.y_max is None:
            self.y_min, self.y_max = y_min, y_max
        if self.zoom_mode == "adaptive":
            self.y_min = min(y_min, 0.9 * self.y_min + 0.1 * y_min)
            self.y_max = max(y_max, 0.9 * self.y_max + 0.1 * y_max)
        elif self.zoom_mode == "keep":
            self.y_min = min(y_min, self.y_min)
            self.y_max = max(y_max, self.y_max)
        else:
            self.y_min = y_min
            self.y_max = y_max
        self.axis.set_ylim([self.y_min, self.y_max])

        if self.blit:
            self.figure.canvas.restore_region(self._background)
            for line in self._lines:
                self.axis.draw_artist(line)
            for surface in self._surfaces:
                self.axis.draw_artist(surface)
            self.figure.canvas.blit(self.axis.bbox)
            self.figure.canvas.flush_events()

        else:
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

    def reset(self):
        self._lines = None

    def fill_between(self, x: np.ndarray, y1: np.ndarray, y2: np.ndarray, *args, **kwargs):
        self.axis.collections.clear()
        self._surfaces = []
        
        x, y1, y2 = np.array(x), np.array(y1), np.array(y2)
        assert x.ndim == 1, "x_data must be a one-dimensional array"
        if y1.ndim == 1:
            y1 = y1[..., None]
        assert y1.ndim == 2, "y1 can be either one or two dimensional!"
        if y2.ndim == 1:
            y2 = y2[..., None]
        assert y2.ndim == 2, "y2 can be either one or two dimensional!"
        
        for i, (y1_, y2_) in enumerate(zip(np.rollaxis(y1, 1), np.rollaxis(y2, 1))):
            self._surfaces.append(self.axis.fill_between(x, y1_, y2_, color=f"C{i}", alpha=0.5, *args, **kwargs))

        self._update()



class InteractiveImagePlot():
    """Class which allows interactive image plots, i.e. image plots
    which can be updated.
    """
    def __init__(self) -> None:
        self._render_figure, self._render_axis = plt.subplots(1)
        self._image = None

    def set_image(self, image: np.ndarray, *args, **kwargs):
        """Set displayed image.

        Parameters
        ----------
        image : np.ndarray
            Image array
        """
        if self._image is None:
            self._image = self._render_axis.imshow(image, *args, **kwargs)
        self._image.set_data(image)
        self._render_figure.canvas.draw()
        self._render_figure.canvas.flush_events()
