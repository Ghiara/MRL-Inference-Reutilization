"""
ToyGoal environment: Simple movement in directions with tasks represented by goal
states.

This module contains different versions of the Toy environment:
- Toy1D
- Toy1dDynamic
- Toy1dContinuous

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-07
"""

import numpy as np
import pygame
import gym
from scipy.linalg import expm
from typing import Dict, Tuple, Any, List
import random
import torch

from .base import ToyEnv, Task

# Space limitations (only defaults)
DELTA_T = 0.1

# Rendering
BLACK = (0, 0, 0)
GREEN   = (0, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 200, 0)
GREY = (127, 127, 127)

AGENT_SIZE = 20
TASK_SIZE = 30
ACTION_FACTOR = 100.0

pygame.font.init()
# my_font = pygame.font.SysFont('Comic Sans MS', 20)
my_font = pygame.font.SysFont('Consolas', 12)


class Toy1D(ToyEnv):
    """Toy goal environment in one dimension. 

    The observation space is a one dimensional line ranging from ``min_pos``
    to ``max_pos``. The action space is a one dimensional line from ``-max_action``
    to ``max_action``. 

    The state transition function is given by 
        next_state = current_state + action

    The reward is computed as
        reward = |current_state - goal|
    where the goal is defined by the task. The agent has several train tasks
    and several test tasks.

    Parameters
    ----------
    n_train_tasks : int
        Number of train tasks.
    n_eval_tasks : int
        Number of test tasks.
    task_generation_mode : str, optional
        Determines how tasks are generated:
        | ``'random'`` | ``'fixed'`` |, by default 'random'
        ``'random'``: Goals are sampled uniformly from the observation space.
        ``'fixed'``: Goals are placed equally distanced between min_pos and max_pos
    one_sided_tasks : bool, optional
        Set to True to map all goal positions to the positive axis.
        By default False
    task_scale : float, optional
        Factor with which sampled goal positions are multiplied. 
        Helps to increase spread of goal positions. 
        By default 1.0
    change_steps : int, optional
        Number of steps until which the task can change, by default 100
    change_prob : float, optional
        Probability of a task change (after ``change_steps``), by default 1.0
    min_pos : float, optional
        Left boundary of the environment. Set to ``-np.inf`` for no boundary. 
        By default -1.0
    max_pos : float, optional
        Right boundary of the environment. Set to ``np.inf`` for no boundary.
        By default 1.0
    max_action : float, optional
        Maximum (absolute) action value, by default 0.1
    """
    # Rendering arguments
    screen_width = 1000
    screen_height = 400

    def __init__(
        self,
        n_train_tasks: int,
        n_eval_tasks: int,
        task_generation_mode: str = 'random',
        one_sided_tasks: bool = False,
        task_scale: float = 1.0,
        change_steps: int = 100,
        change_prob: float = 1.0,
        min_pos: float = -1.0,
        max_pos: float = 1.0,
        max_action: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.max_action = max_action

        self.observation_space = gym.spaces.Box(low=self.min_pos, high=self.max_pos, shape=(1,))
        self.action_space = gym.spaces.Box(low=-self.max_action, high=self.max_action, shape=(1,))
        self.state = np.zeros((1))

        train_tasks, eval_tasks = self._init_tasks(
            n_train_tasks, n_eval_tasks, 
            mode=task_generation_mode, 
            one_sided=one_sided_tasks,
            task_scale=task_scale,
        )
        super().__init__(
            train_tasks, eval_tasks,
            change_steps=change_steps, 
            change_prob=change_prob, 
            task_scale=task_scale,
            *args, **kwargs
        )

        # Pygame rendering ...
        self._last_action = 0.0
        self._last_clipped_action = 0.0
        self._last_state = self.state
        self._render_borders = (self.min_pos, self.max_pos) # keeps track of rendering borders
        self._border_padding = 2 * max_action if max_action < np.inf else 100.0        # Padding for rendering of infinite environments
        self._xticks = None                                 # x-ticks for the coordinate axis

    @property
    def observation(self):
        return self.state

    def _init_tasks(
            self,
            n_train_tasks: int, 
            n_eval_tasks: int, 
            mode: str = 'random', 
            one_sided: bool = False,
            task_scale: float = 1.0,
        ) -> Tuple[List[Task], List[Task]]:
        """
        Sample training and evaluation task sets.
        """
        if mode == 'random':
            train_goals = [self.observation_space.sample() * task_scale for _ in range(n_train_tasks)]
            test_goals = [self.observation_space.sample() * task_scale for _ in range(n_eval_tasks)]
        elif mode == 'fixed':
            low = self.min_pos if self.min_pos != -np.inf else - (n_train_tasks * self.max_action)
            high = self.max_pos if self.max_pos != np.inf else (n_train_tasks * self.max_action)
            train_goals = np.linspace(low, high, n_train_tasks) * task_scale
            test_goals = np.linspace(low, high, n_eval_tasks) * task_scale
        else:
            raise ValueError(f'Unknown option \'{mode}\' for argument \'task_generation_mode\'.')
        
        if one_sided:
            train_goals = [np.abs(train_goal) for train_goal in train_goals]
            test_goals = [np.abs(test_goal) for test_goal in test_goals]
        
        train_tasks = []
        eval_tasks = []
        next_id = 0
        for goal in train_goals:
            train_tasks.append({'id': next_id, 'goal': np.array([goal])})
            next_id += 1
        for goal in test_goals:
            eval_tasks.append({'id': next_id, 'goal': np.array([goal])})
            next_id += 1
        return train_tasks, eval_tasks

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self._try_task_update()
        if isinstance(action, torch.Tensor):
            clipped_action = action.clip(-self.max_action, self.max_action).cpu().numpy()
            clipped_action = np.array([clipped_action]) 
        else:
            clipped_action = action.clip(-self.max_action, self.max_action)
        assert self.action_space.contains(clipped_action.astype(np.float32)), "Clipped action is not in action space!"

        self._last_state = self.state
        self.state = (self.state + clipped_action).clip(self.min_pos, self.max_pos)
        assert self.observation_space.contains(self.state.astype(np.float32)), "State is not in state space!"
        reward = - float(np.abs(self.state - self._task['goal']))
        done = False
        env_info = {
            'true_task': self._task
        }

        self._steps_since_task_update += 1

        self._last_action = action.item()
        self._last_clipped_action = clipped_action.item()

        return self.state, reward, done, False, env_info

    def _draw_env(self):

        # Determine rendering boundaries
        min_pos, max_pos = self._render_borders
        if self.observation_space.is_bounded():
            pass    # Bounded environment -> Use the environment boundaries
        else:
            # Adapt rendered range to position of agent
            window_size = 5 * self._border_padding
            if min_pos > self.state[0] - self._border_padding or max_pos == np.inf:
                min_pos = self.state[0] - self._border_padding
                max_pos = min_pos + window_size
            if max_pos < self.state[0] + self._border_padding or min_pos == -np.inf:
                max_pos = self.state[0] + self._border_padding
                min_pos = max_pos - window_size
        self._render_borders = (min_pos, max_pos)
        eps = 1e-2 * (max_pos - min_pos)
        coordinate = lambda x: (x - min_pos + eps)/(max_pos - min_pos + 2*eps)*self.screen_width
        
        # Coordinate axis
        pygame.draw.line(
            self._screen,
            GREY,
            start_pos = [0, self.screen_height/2],
            end_pos=[self.screen_width, self.screen_height/2],
            width=2,
        )
        # x-ticks
        range = max_pos - min_pos
        n_ticks = 10
        scale = int(np.floor(np.log10(range/n_ticks)))   # potency of 10
        tick_dist = max(10 ** scale, range / 20)
        if self._xticks is None or self._xticks[0] > min_pos or self._xticks[-1] < max_pos:
            # Recomputation only if needed, make xticks larger then rendering --> smoother animation
            min_tick = np.round(min_pos - 5 * self._border_padding - tick_dist, -scale)
            max_tick = np.round(max_pos + 5 * self._border_padding + tick_dist, -scale)
            self._xticks = np.arange(min_tick, max_tick, tick_dist)
        for x in self._xticks:
            pygame.draw.line(
                self._screen,
                BLACK,
                start_pos = [coordinate(x), 0.97*(self.screen_height/2)],
                end_pos = [coordinate(x), 1.03*(self.screen_height/2)],
                width=1,
            )
            label = my_font.render(f"{x:.1f}", True, GREY)
            self._screen.blit(label, (coordinate(x)-2, 1.05*(self.screen_height/2)))

        # Environment
        pygame.draw.rect(# Goal
            self._screen,
            GREEN,
            [coordinate(float(self._task['goal'])) - TASK_SIZE/2,
            self.screen_height/2 - TASK_SIZE/2, TASK_SIZE, TASK_SIZE]
        )
        pygame.draw.rect(# Agent
            self._screen,
            BLACK,
            [coordinate(float(self._last_state[0])) - AGENT_SIZE/2,
            self.screen_height/2 - AGENT_SIZE/2, AGENT_SIZE, AGENT_SIZE]
        )
        pygame.draw.line(# Action
            self._screen,
            ORANGE,
            start_pos = [coordinate(float(self._last_state[0])), self.screen_height/2],
            end_pos = [coordinate(float(self._last_state[0])) + ACTION_FACTOR * self._last_action / tick_dist, self.screen_height/2],
            width=8,
        )
        pygame.draw.line(# Clipped action
            self._screen,
            RED,
            start_pos = [coordinate(float(self._last_state[0])), self.screen_height/2],
            end_pos = [coordinate(float(self._last_state[0])) + ACTION_FACTOR * self._last_clipped_action / tick_dist, self.screen_height/2],
            width=8,
        )
        

class Toy1dDynamic(Toy1D):
    """Toy goal environment in one dimension with dynamic equations
                x'' = a

    The observation space is a one dimensional line ranging from ``self.min_pos``
    to ``self.max_pos``. The action space is are accelerations from ``MIN_ACC``
    to ``MAX_ACC``. 

    The reward is computed as
        reward = |current_state - goal|.
    The agent has several train tasks and several test tasks.

    Parameters
    ----------
    n_train_tasks : int
        Number of train tasks.
    n_eval_tasks : int
        Number of test tasks.
    task_generation_mode : str, optional
        Determines how tasks are generated:
        | ``'random'`` | ``'fixed'`` |, by default 'random'
        ``'random'``: Goals are sampled uniformly from the observation space.
        ``'fixed'``: Goal is placed at ``MAX_STATE`` (for testing purposes only)
    change_steps : int, optional
        Number of steps until which the task can change, by default 100
    change_prob : float, optional
        Probability of a task change (after ``change_steps``), by default 1.0
    min_pos : float, optional
        Left boundary of the environment. Set to ``-np.inf`` for no boundary. 
        By default -1.0
    max_pos : float, optional
        Right boundary of the environment. Set to ``np.inf`` for no boundary.
        By default 1.0
    max_action : float, optional
        Maximum (absolute) action value, by default 0.1
    max_vel : float, optional
        Maximum (absolute) velocity value, by default 1.0
    """
    def __init__(
        self,
        n_train_tasks: int,
        n_eval_tasks: int,
        task_generation_mode: str = 'random',
        change_steps: int = 100,
        change_prob: float = 1.0,
        min_pos: float = -1.0,
        max_pos: float = 1.0,
        max_action: float = 0.1,
        max_vel: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            n_train_tasks, n_eval_tasks, task_generation_mode, 
            change_steps=change_steps, change_prob=change_prob, 
            min_pos=min_pos, max_pos=max_pos, max_action=max_action, 
            *args, **kwargs
        )
        self.max_vel = max_vel
        self.state = np.zeros((2))
        self._border_reached = False

        self.observation_space = gym.spaces.Box(low=np.array([self.min_pos, -self.max_vel]), high=np.array([self.max_pos, self.max_vel]))
        self.action_space = gym.spaces.Box(low=-self.max_action, high=self.max_action, shape=(1,))

        # Rendering
        self._border_padding = max_vel if max_vel < np.inf else 100.0
 
    def reset(self) -> Any:
        _, info = super().reset()
        self.state[1] = 0
        return self.state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self._try_task_update()

        acceleration = action.clip(-self.max_action, self.max_action)
        assert self.action_space.contains(acceleration.astype(np.float32)), "Clipped action is not in action space!"

        self._last_state = self.state
        self.state += np.array([self.state[1], acceleration.item()]) * DELTA_T
        self.state = np.clip(self.state, a_min=[self.min_pos, -self.max_vel], a_max=[self.max_pos, self.max_vel])
        if (self.state[0] == self.min_pos or self.state[0] == self.max_pos) and not self._border_reached:
            self.state[1] = 0
            self._border_reached = True
        else:
            self._border_reached = False
        assert self.observation_space.contains(self.state.astype(np.float32)), "State is not in state space!"

        reward = - float(np.abs(self.state[0] - self._task['goal']))
        done = False
        env_info = {
            'true_task': self._task
        }

        self._steps_since_task_update += 1
        self._last_clipped_action = acceleration.item()
        self._last_action = action.item()

        return self.state, reward, done, False, env_info


class Toy1dContinuous(Toy1dDynamic):
    """Toy environment with discretized continuous dynamics of a Mass-Spring-Damper system:
        x_1: Position, x_2: Velocity \n
        x_1' = x_2 \n
        x_2' = - k/m * x_1 - d/m * x_2 + 1/m * u \n
    where m is the mass, k is the spring constant, d is the damping factor,
    and u is the force applied to the mass.

    The reward is computed as
        reward = |current_state - goal|.
    The agent has ``n_train_tasks`` train tasks and ``n_eval_tasks`` test tasks.

    An introduction to discretization is given here: https://en.wikipedia.org/wiki/Discretization

    Discretization is performed w.r.t. the time interval constant DELTA_T.

    Parameters
    ----------
    n_train_tasks : int
        Number of tasks for training
    n_eval_tasks : int
        Number of tasks for testing
    task_generation_mode : str, optional
        Determines how tasks are generated:
        | ``'random'`` | ``'fixed'`` |, by default 'random'
        ``'random'``: Goals are sampled uniformly from the observation space.
        ``'fixed'``: Goal is placed at ``MAX_STATE`` (for testing purposes only)
    change_steps : int, optional
        Number of steps until which the task can change, by default 100
    change_prob : float, optional
        Probability of a task change (after ``change_steps``), by default 1.0
    mass : float, optional
        Mass m, by default 1.0
    spring_constant : float, optional
        Spring constant k, by default 0.0
    damping : float, optional
        Damping factor d, by default 0.0
    min_pos : float, optional
        Left boundary of the environment. Set to ``-np.inf`` for no boundary. 
        By default -1.0
    max_pos : float, optional
        Right boundary of the environment. Set to ``np.inf`` for no boundary.
        By default 1.0
    max_action : float, optional
        Maximum (absolute) action value, by default 0.1
    max_vel : float, optional
        Maximum (absolute) velocity value, by default 1.0
    """
    render_fps = 60
    def __init__(
            self,
            n_train_tasks: int,
            n_eval_tasks: int,
            task_generation_mode: str = 'random',
            change_steps: int = 100,
            change_prob: float = 1.0,
            mass: float = 1.0,
            spring_constant: float = 0.0,
            damping: float = 0.0,
            min_pos: float = -1.0,
            max_pos: float = 1.0,
            max_action: float = 0.1,
            max_vel: float = 1.0,
            *args, 
            **kwargs,
        ) -> None:
        super().__init__(
            n_train_tasks, n_eval_tasks, task_generation_mode, 
            change_steps, change_prob, min_pos=min_pos, 
            max_pos=max_pos, max_action=max_action, max_vel=max_vel,
            *args, **kwargs
        )

        # Continuous dynamics matrices: x' = Ax + Bu where x is the state vector and u is the input
        A = np.array([[0, 1],[-spring_constant/mass, -damping/mass]])
        B = np.array([0, 1/mass])

        # Discretized dynamics matrices
        # A_d = e^{At}, B_d = A^{-1} (A_d - I) B
        # Discrete dynamics: x[k+1] = A_d x[k] + B_d u[k] where x is the state vector and u is the input
        self.A = expm(A * DELTA_T)
        if np.linalg.det(A) == 0:
            A_inv = np.linalg.pinv(A)
            self.B = A_inv @ (self.A - np.eye(*self.A.shape)) @ B
        else: 
            self.B = np.linalg.solve(A, (self.A - np.eye(*self.A.shape)) @ B)
        self.B = self.B.reshape(2,1)


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self._try_task_update()

        force = action.clip(-self.max_action, self.max_action)
        assert self.action_space.contains(force.astype(np.float32)), "Clipped action is not in action space!"

        self._last_state = self.state
        self.state = self.A @ self.state + self.B @ force
        self.state = np.clip(self.state, a_min=[self.min_pos, -self.max_vel], a_max=[self.max_pos, self.max_vel])
        if (self.state[0] == self.min_pos or self.state[0] == self.max_pos) and not self._border_reached:
            self.state[1] = 0
            self._border_reached = True
        else:
            self._border_reached = False
        assert self.observation_space.contains(self.state.astype(np.float32)), "State is not in state space!"

        reward = - float(np.abs(self.state[0] - self._task['goal']))
        done = False
        env_info = {
            'true_task': self._task
        }

        self._steps_since_task_update += 1
        self._last_action = action.item()
        self._last_clipped_action = force.item()

        return self.state, reward, done, False, env_info


Toy1dDiscretized = Toy1dContinuous  # Legacy support


# class Toy1dRand(Toy1D):
#     def __init__(
#             self,
#             n_train_tasks: int,
#             n_eval_tasks: int,
#             task_generation_mode: str = 'random',
#             one_sided_tasks: bool = False,
#             task_scale: float = 1.0,
#             change_steps: int = 100,
#             change_prob: float = 1.0,
#             min_pos: float = -1.0,
#             max_pos: float = 1.0,
#             max_action: float = 0.1,
#             max_multiplier: float = 2.0,
#             min_multiplier: float = 0.5,
#             *args,
#             **kwargs,
#             ) -> None:
#         super().__init__(
#             n_train_tasks, n_eval_tasks, task_generation_mode, 
#             change_steps=change_steps, change_prob=change_prob, 
#             min_pos=min_pos, max_pos=max_pos, max_action=max_action, 
#             *args, **kwargs
#         )
#         self.max_multiplier = max_multiplier
#         self.min_multiplier = min_multiplier

#         def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
#             self._try_task_update()

#             clipped_action = action.clip(-self.max_action, self.max_action)
#             assert self.action_space.contains(clipped_action.astype(np.float32)), "Clipped action is not in action space!"

#             self._last_state = self.state
#             self.state = (self.state + clipped_action).clip(self.min_pos, self.max_pos)
#             assert self.observation_space.contains(self.state.astype(np.float32)), "State is not in state space!"
#             reward = - float(np.abs(self.state - self._task['goal']))
#             done = False
#             env_info = {
#                 'true_task': self._task
#             }

#             self._steps_since_task_update += 1

#             self._last_action = action.item()
#             self._last_clipped_action = clipped_action.item()

#             return self.state, reward, done, False, env_info
