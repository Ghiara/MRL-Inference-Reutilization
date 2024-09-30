import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import Union, List, Dict
array_like = Union[np.ndarray, List[float]]



class Plot():
    def __init__(
        self,
        x: str,
        y: Union[str, List[str]],
        title: str = None,
        x_label: str = None,
        y_label: Union[str, List[str]] = None,
        data: Union[Dict[str, array_like], pd.DataFrame] = None,
    ) -> None:

        # Parse inputs
        if not isinstance(y, list): y = [y]
        x_label = x_label if x_label is not None else x
        y_label = y_label if y_label is not None else y
        if not isinstance(y_label, list): y_label = [y_label]

        self.x = x
        self.ys = y
        self.x_label = x_label
        self.y_labels = y_label

        # Initialize figure and lines
        self.figure, self.axs = plt.subplots(1)
        self.lines = []
        for y, y_label in zip(self.ys, self.y_labels):
            self.lines.append(self.axs.plot([],[], label=y_label)[0])

        # Initialize axes and figure information
        self.axs.set_xlabel(x_label)
        self.axs.legend()
        if title is not None:
            self.figure.suptitle(title)

        if data is not None:
            self.update(data)

    def update(self, data: pd.DataFrame):
        for y, line in zip(self.ys, self.lines):
            line.set_data(data[self.x], data[y])

        self.axs.relim()
        self.axs.autoscale_view(True,True,True)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()