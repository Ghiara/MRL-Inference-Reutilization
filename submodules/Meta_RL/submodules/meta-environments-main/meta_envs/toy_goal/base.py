"""
ToyGoal environment: Simple movement in directions with tasks represented by goal
states.

This module contains the base class for Toy environments: ``ToyEnv``

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-03-06
"""

import sys
import gym
import pygame
import numpy as np
from typing import Any, Dict, List, Tuple
from abc import ABC, abstractmethod
import traceback

from ..base import TaskSetEnvironment, pygame_coordinates_to_image_coordinates, Task

class ToyEnv(TaskSetEnvironment, ABC):
    """Base class for Toy environments which implements important properties
    and base functionalities (including rendering!).

    Parameters
    ----------
    train_tasks : List[Task]
        Set of training tasks
    eval_tasks : List[Task]
        Set of evaluation tasks
    change_steps : int, optional
        Number of steps until which the task can change, by default 100
    change_prob : float, optional
        Probability of a task change (after ``change_steps``), by default 1.0
    reset_mode : str, optional
        State reset mode, can be one of:
        ``'zero'``: Reset position and velocity to zero
        ``'random'``: Reset position to a random value from the observation space, velocity to zero
        ``'stay'``: Do not reset position (only task)
    task_scale : float, optional
        Scale of the tasks. Used for randomized resets to ensure that reset
        positions scale like tasks.
        By default 1.0
    """
    # Rendering arguments
    render_fps = 30 # Frames per second during rendering
    screen_width = 400     # Width of the frame
    screen_height = 400    # Height of the frame

    def __init__(
        self,
        train_tasks: List[Task],
        eval_tasks: List[Task],
        change_steps: int = 100,
        change_prob: float = 1.0,
        reset_mode: str = "zero",
        task_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        
        super().__init__(
            train_tasks, eval_tasks, 
            change_steps=change_steps, 
            change_prob=change_prob,
            *args, **kwargs
        )

        assert reset_mode in ("zero", "random", "stay"), f"Unknown option {reset_mode} for argument reset_mode."
        self.reset_mode = reset_mode
        self.task_scale = task_scale

        self.observation_space: gym.spaces.Box  # For typing
        self.action_space: gym.spaces.Box       # For typing

        # Rendering
        self._screen = None

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.sample_task()
        if not hasattr(self, 'reset_mode'):
            # TODO: remove in future commit, this is legacy support
            self.reset_mode = "zero"
        if self.reset_mode == "zero":
            self.state = np.zeros(self.observation_space.shape[0])
        elif self.reset_mode == "random":
            self.state = self.observation_space.sample() * self.task_scale
            self.state = np.clip(self.state, a_min=self.observation_space.low, a_max=self.observation_space.high)
        elif self.reset_mode == "stay":
            pass
        else:
            raise ValueError(f"Value {self.reset_mode} is not a valid option for self.reset_mode.")
        return self.state, {}

    def render(self, mode: str = 'human', width: int = None, height: int = None):
        if width is not None: self.screen_width = width
        if height is not None: self.screen_height = height
        if mode == "rgb_array":
            import os
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        try:

            if self._screen is None:
                pygame.init()
                self._clock = pygame.time.Clock()
                pygame.display.set_caption("ToyEnv")
                print(f"Width: {self.screen_width}, Height: {self.screen_height}")
                self._screen = pygame.display.set_mode((self.screen_width,self.screen_height))
                self._screen.fill("white")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self._screen = None
                    sys.exit(0)

            self._screen.fill("white")

            self._draw_env()

            pygame.display.flip()
            if not mode == "rgb_array":
                self._clock.tick(self.render_fps)

            image = pygame.surfarray.pixels3d(self._screen).copy()
            return pygame_coordinates_to_image_coordinates(image)

        except: # Make sure that window is closed if error occurs
            pygame.quit()
            self._screen = None
            traceback.print_exc()
            sys.exit()

    def get_image(self, width: int = 800, height: int = 400):
        return self.render(mode='rgb_array', width=width, height=height)

    @abstractmethod
    def _draw_env(self):
        """ This function draws the environment. Called by ``render()``."""
        raise NotImplementedError
