"""
This module contains the an abstract base class for meta / multitask environments.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-07
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import gym
import numpy as np
import random


Task = Dict[str, Any]


class MultiTaskEnv(gym.Env, ABC):
    """Abstract base class for multi-task environments.
    """

    @property
    @abstractmethod
    def task_encoding_shape(self) -> Tuple[int, ...]:
        """The size of the task encodings returned by ``step()`` and ``reset()``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def task(self) -> Task:
        """The current task as a dictionary. 
        
        This might not be identical to the task encodings returned by 
        ``step()`` and ``reset()`` but can be understood as a human-readable
        task description.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_task(self) -> Task:
        """Set the current task to a random task from the set/distribution of tasks.

        Returns
        -------
        Task
            The new task
        """
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool, bool, dict]:
        """Environment step of the multi-task environment. Additonally to the
        observation, this environment also returns a task encoding in form
        of a numpy array.

        Parameters
        ----------
        action : np.ndarray
            Action

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, bool, bool, dict]
            Observation,
            Task encoding,
            Reward,
            Terminal indicator,
            Truncation indicator,
            Environment information
        """
        return super().step(action)
    
    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Resets the environment. Additionally to the current observation,
        this environment also returns a task encoding in form of a numpy array.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, dict]
            Observation
            Task encoding
            Environment information
        """
        return super().reset(*args, **kwargs)


class MetaEnv(gym.Env, ABC):
    """Abstract base class for meta environments.
    """
    
    # Rendering arguments
    render_fps = 30 # Frames per second during rendering
    screen_width = 400     # Width of the frame
    screen_height = 400    # Height of the frame

    def __init__(self, *args, **kwargs) -> None:
        self.meta_mode = 'train'
        super().__init__(*args, **kwargs)

    def set_meta_mode(self, mode='train'):
        """Set the environment to 'train' or 'test'.

        This changes the set of tasks from which the current task can be sampled.
        There are separate sets of train tasks and test tasks. 

        Parameters
        ----------
        mode : str, optional
            The meta mode, can either be 'train' or 'test', by default 'train'
        """
        assert (mode in ['train', 'test']), "Argument `mode` can only be set to 'train' or 'test'."
        self.meta_mode = mode

    @property
    @abstractmethod
    def task(self) -> Task:
        """The current task description as a dictionary.

        The dictionary must contain the key 'id' which identifies the task 
        (could be an integer or a string).
        """
        raise NotImplementedError

    @abstractmethod
    def sample_task(self) -> Task:
        """Set the current task to a random task from the set/distribution of tasks.

        Returns
        -------
        Task
            The new task
        """
        raise NotImplementedError

    # Optional, useful additional functions (compatibility)

    @property
    def tasks(self) -> Any:
        """ 
        The set of tasks (train and test)

        Note: This property increases compatibility with legacy code.
        """
        raise NotImplementedError
    
    def get_all_task_idx(self):
        """
        Get all indexes of the tasks (usually train tasks and eval tasks).

        Note: This method increases compatibility with legacy code.
        """
        raise NotImplementedError

    def reset_task(self, idx: int, *args, **kwargs):
        """
        Reset the task to the task with a given index.

        Note: This method increases compatibility with legacy code.

        Parameters
        ----------
        idx : int
            Index of the task which is chosen from the list of train and eval tasks
        """
        raise NotImplementedError

    def set_task(self, task_nr: int):
        """
        Set the task to the task with a given index.

        Note: This method increases compatibility with legacy code.

        Parameters
        ----------
        idx : int
            Index of the task which is chosen from the list of train and eval tasks
        """
        raise NotImplementedError

    def get_image(self, width: int, height: int):
        """
        Get an image of the current state of the environment (similar to 
        ``render(mode='rgb_array')``).

        Note: This method increases compatibility with legacy code.

        Parameters
        ----------
        width : int
            Image width (pixels)
        height : int
            Image height (pixels)
        """
        raise NotImplementedError

    def clear_buffer(self):
        """
        Clear buffer.

        Note: This method increases compatibility with legacy code.
        """
        pass


class TaskSetEnvironment(MetaEnv, ABC):
    """A meta environment which uses a (potentially fixed) set of training and
    evaluation tasks from which it selects its current task.

    Tasks are updated after a minimum of ``change_steps`` steps with probability
    ``change_prob``. See ``_try_task_update()``.

    Parameters
    ----------
    train_tasks : List[Task]
        List of training tasks
    test_tasks : List[Task]
        List of evaluation tasks
    change_steps : int
        Minimum number of steps required before a task update
    change_prob : float
        Probability of changing the task in each steps (if step > change_steps)
    """
    def __init__(self, train_tasks: List[Task], eval_tasks: List[Task], change_steps: int, change_prob: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self._task: Task = train_tasks[0]

        self.change_steps = change_steps
        self.change_prob = change_prob

        self._steps_since_task_update = 0
        self.sample_task()

    @property
    def task(self) -> Task:
        # You can overwrite this method in derived classes for additional
        # control
        return self._task
    
    @task.setter
    def task(self, task: Task):
        # You can overwrite this method in derived classes for additional
        # control
        self._task = task

    @property
    def tasks(self) -> Any:
        return self.train_tasks + self.eval_tasks
    
    def sample_task(self):
        tasks = self.train_tasks if self.meta_mode == 'train' else self.eval_tasks
        self._task = random.choice(tasks)
        self._steps_since_task_update = 0

    def _try_task_update(self):
        """
        Try a task update based on the steps since the last task update and
        task update probability.

        NOTE: You need to increment ``self._steps_since_task_update`` on your 
        own, e.g. in ``step()``.
        """
        if self._steps_since_task_update >= self.change_steps \
            and random.random() <= self.change_prob:
            self.sample_task()

    def get_all_task_idx(self):
        return range(len(self.train_tasks) + len(self.eval_tasks))

    def reset_task(self, idx, *args, **kwargs):
        self._task = (self.train_tasks + self.eval_tasks)[idx]
        self.reset()

    def set_task(self, task_nr: int):
        self._task = (self.train_tasks + self.eval_tasks)[task_nr]


def pygame_coordinates_to_image_coordinates(image: np.ndarray) -> np.ndarray:
    """Change the coordinate system of a pygame surface array to image coordinates:

    surface coordinates: 
    ```
     x --- rows --->
     |
     |
    columns
     |
     |
     v
    ```
    image coordinates: 
    ```
     x --- columns --->
     |
     |
    rows
     |
     |
     v
    ```

    Parameters
    ----------
    image : np.ndarray
        Image in surface coordinates (from pygame)

    Returns
    -------
    np.ndarray
        Image in image coordinates
    """
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

def coordinates_to_pygame_coordinates(coordinates: np.ndarray, y_max: int) -> np.ndarray:
    """Change the coordinate system to pygame coordinates:

    coordinates (coordinate system in bottom left corner): 
    ```
     ^
     |
     |
    columns
     |
     |
     x --- rows --->
    ```

    pygame coordinates: 
    ```
     x --- rows --->
     |
     |
    columns
     |
     |
     v
    ```
    

    Parameters
    ----------
    coordinates : np.ndarray
        Coordinates, shape (2)
    y_max : int
        Screen height (in pixels)

    Returns
    -------
    np.ndarray
        Image in image coordinates
    """
    coordinates[1] = y_max - coordinates[1]
    return coordinates