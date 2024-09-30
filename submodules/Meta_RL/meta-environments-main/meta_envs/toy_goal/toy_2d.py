"""
ToyGoal environment: Simple movement in directions with tasks represented by goal
states.

This module contains different versions of the Toy environment:
- Toy1D
- Toy1dDynamic
- Toy1dDiscretized

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-20
"""

import numpy as np
import copy
import pygame
import gym
from scipy.linalg import expm
from typing import Dict, Tuple, Any, List

from ..base import coordinates_to_pygame_coordinates

from .base import ToyEnv, Task

# Space limitations
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


class Toy2D(ToyEnv):
    """Toy goal environment in two dimensions. 

    The observation space is a two dimensional plane ranging from ``min_state``
    to ``max_state`` (in both directions). 
    The action space contains two-dimensional inputs with a maximum absolute
    value of ``max_action``.

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
        ``'fixed'``: Goals are distributed linearly between min_pos and max_pos,
                    if min_pos or max_pos are infinite, tasks are placed between
                    -task_scale and +task_scale
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
    reward_type : str
        The type of norm which is used for reward computation, can be any of 
        | ``'L1'`` | ``'L2'`` |
    min_pos : float, optional
        Left and bottom border of the environment, by default -1.0
    max_pos : float, optional
        Right and top border of the environment, by default 1.0
    max_action : float, optional
        Maximum norm of the action, by default 0.1
    """
    # Rendering arguments
    screen_width = 1000
    screen_height = 1000

    def __init__(
        self,
        n_train_tasks: int,
        n_eval_tasks: int,
        task_generation_mode: str = 'random',
        one_sided_tasks: bool = False,
        task_scale: float = 1.0,
        change_steps: int = 100,
        change_prob: float = 1.0,
        reward_type: str = "L2",
        min_pos: float = -1.0,
        max_pos: float = 1.0,
        max_action: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        self.max_pos_x = max_pos
        self.max_pos_y = max_pos
        self.min_pos_x = min_pos
        self.min_pos_y = min_pos
        self.max_action = max_action

        self.observation_space = gym.spaces.Box(
            low=np.array([min_pos, min_pos]), 
            high=np.array([max_pos, max_pos]), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-max_action, -max_action]), 
            high=np.array([max_action, max_action]), 
            dtype=np.float32
        )
        self.state = np.zeros((1))
        
        train_tasks, eval_tasks = self._init_tasks(
            n_train_tasks, n_eval_tasks, 
            mode=task_generation_mode,
            task_scale=task_scale,
            one_sided=one_sided_tasks,
        )
        
        super().__init__(
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            change_steps=change_steps, 
            change_prob=change_prob, 
            task_scale=task_scale,
            *args, **kwargs
        )

        assert reward_type in ("L1", "L2"), f"Unknown reward type {reward_type}."
        self._reward_type = reward_type

        # Pygame rendering ...
        self._last_state = self.state
        self._last_action = np.array([0.0, 0.0])
        self._last_clipped_action = np.array([0.0, 0.0])
        self._render_borders = (self.min_pos_x, self.min_pos_y, self.max_pos_x, self.max_pos_y) # keeps track of rendering borders
        self._border_padding = 10 * max_action if max_action < np.inf else 100.0        # Padding for rendering of infinite environments
        self._xticks = None             # x-ticks for the coordinate axis
        self._yticks = None             # y-ticks for the corrdinate axis
        self._coordinate_origin = None  # coordinate system origin (in world coordinates)

    def _init_tasks(
            self, 
            n_train_tasks: int, 
            n_eval_tasks: int,
            mode: str = 'random',
            one_sided: bool = False,
            task_scale: float = 1.0,
        ) -> Tuple[List[Task], List[Task]]:
        """Initialize task sets.

        Parameters
        ----------
        n_train_tasks : int
            Number of train tasks
        n_eval_tasks : int
            Number of test tasks
        mode : str, optional
            Generation mode, can be one of
            ``'random'`` | ``'fixed'``,
            by default ``'random'``
        one_sided : bool, optional
            Set to True to ensure that all goals are in sector I.
            By default False
        task_scale: float, optional
            Factor with which goal positions are multiplied. By default 1.0
        """
        if mode == 'random':
            train_goals = [self.observation_space.sample() * task_scale for _ in range(n_train_tasks)]
            test_goals = [self.observation_space.sample() * task_scale for _ in range(n_eval_tasks)]
        elif mode == 'fixed':
            min_x = self.min_pos_x if self.min_pos_x != -np.inf else -task_scale
            max_x = self.max_pos_x if self.max_pos_x != np.inf else task_scale
            min_y = self.min_pos_y if self.min_pos_y != -np.inf else -task_scale
            max_y = self.max_pos_y if self.max_pos_y != np.inf else task_scale
            grid_len = int(np.ceil(np.sqrt(n_train_tasks)))
            x = np.linspace(min_x, max_x, grid_len)
            y = np.linspace(min_y, max_y, grid_len)
            xv, yv = np.meshgrid(x, y)
            train_goals = np.concatenate([xv[..., None], yv[..., None]], axis=-1)
            test_goals = np.concatenate([xv[..., None], yv[..., None]], axis=-1)
            train_goals = train_goals.reshape([grid_len**2, -1])
            test_goals = test_goals.reshape([grid_len**2, -1])
        else:
            raise ValueError(f'Unknown option \'{mode}\' for argument \'task_generation_mode\'.')

        if one_sided:
            train_goals = [np.abs(train_goal) for train_goal in train_goals]
            test_goals = [np.abs(test_goal) for test_goal in test_goals]
        
        train_tasks = []
        eval_tasks = []
        next_id = 0
        for goal in train_goals:
            train_tasks.append({'id': next_id, 'goal': goal})
            next_id += 1
        for goal in test_goals:
            eval_tasks.append({'id': next_id, 'goal': goal})
            next_id += 1

        return train_tasks, eval_tasks

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self._try_task_update()

        clipped_action = action * np.min([1.0, self.max_action / np.linalg.norm(action)])
        # assert self.action_space.contains(clipped_action.astype(np.float32)), f"Clipped action ({clipped_action}) is not in action space!"

        self._last_state = self.state
        self.state = (self.state + clipped_action)
        self.state = np.clip(self.state, 
                             a_min = np.array([self.min_pos_x, self.min_pos_y]),
                             a_max=np.array([self.max_pos_x, self.max_pos_y])
                            )
        assert self.observation_space.contains(self.state.astype(np.float32)), "State is not in state space!"
        
        if self._reward_type == "L1":
            reward = - np.linalg.norm(self.state - self._task['goal'], ord=1)
        elif self._reward_type == "L2":
            reward = - np.linalg.norm(self.state - self._task['goal'], ord=2)
        else:
            raise ValueError(f"Unknown option for reward type: {self._reward_type}")

        done = False
        env_info = {
            'true_task': self._task
        }

        self._steps_since_task_update += 1

        self._last_action = action
        self._last_clipped_action = clipped_action

        return self.state, reward, done, False, env_info

    def _draw_env(self):
        """ Environment rendering, called by ``render()``. """

        # Determine rendering boundaries
        min_pos_x, min_pos_y, max_pos_x, max_pos_y = self._render_borders
        if self.observation_space.is_bounded():
            pass    # Bounded environment -> Use the environment boundaries
        else:
            # Adapt rendered range to position of agent
            window_size = 10 * self._border_padding
            if min_pos_x > self.state[0] - self._border_padding or max_pos_x == np.inf:
                min_pos_x = self.state[0] - self._border_padding
                max_pos_x = min_pos_x + window_size
            if max_pos_x < self.state[0] + self._border_padding or min_pos_x == -np.inf:
                max_pos_x = self.state[0] + self._border_padding
                min_pos_x = max_pos_x - window_size
            if min_pos_y > self.state[1] - self._border_padding or max_pos_y == np.inf:
                min_pos_y = self.state[1] - self._border_padding
                max_pos_y = min_pos_y + window_size
            if max_pos_y < self.state[1] + self._border_padding or min_pos_y == -np.inf:
                max_pos_y = self.state[1] + self._border_padding
                min_pos_y = max_pos_y - window_size
        self._render_borders = (min_pos_x, min_pos_y, max_pos_x, max_pos_y)
        eps_x = 3e-2 * (max_pos_x - min_pos_x)
        eps_y = 3e-2 * (max_pos_y - min_pos_y)
        x_range, y_range = max_pos_x - min_pos_x, max_pos_y - min_pos_y

        def coordinate(x):
            # Maps world-coordinates to pygame coordinates for rendering
            normalized = (x - np.array([min_pos_x - eps_x, min_pos_y - eps_y])) \
                        / np.array([x_range + 2*eps_x, y_range + 2*eps_y])
            screen_coords = normalized * np.array([self.screen_width, self.screen_height])
            return coordinates_to_pygame_coordinates(screen_coords, y_max = self.screen_height)


        # Coordinate system
        # origin
        if self._coordinate_origin is None:
            self._coordinate_origin = np.array([min_pos_x, min_pos_y])
        if self._coordinate_origin[0] < min_pos_x:
            self._coordinate_origin[0] = min_pos_x
        if self._coordinate_origin[0] > max_pos_x:
            self._coordinate_origin[0] = max_pos_x
        if self._coordinate_origin[1] < min_pos_y:
            self._coordinate_origin[1] = min_pos_y
        if self._coordinate_origin[1] > max_pos_y:
            self._coordinate_origin[1] = max_pos_y
        screen_origin = coordinate(self._coordinate_origin)
        # x-axis
        pygame.draw.line(
            self._screen,
            GREY,
            start_pos = [0, screen_origin[1]],
            end_pos=[self.screen_width, screen_origin[1]],
            width=2,
        )
        # x-ticks
        n_ticks = 10
        scale = int(np.floor(np.log10(x_range/n_ticks)))   # potency of 10
        x_tick_dist = max(10 ** scale, x_range / 20)
        if self._xticks is None or self._xticks[0] > min_pos_x or self._xticks[-1] < max_pos_x:
            # Recomputation only if needed, make xticks larger then rendering --> smoother animation
            min_tick = np.round(min_pos_x - 5 * self._border_padding - x_tick_dist, -scale)
            max_tick = np.round(max_pos_x + 5 * self._border_padding + x_tick_dist, -scale)
            self._xticks = np.arange(min_tick, max_tick, x_tick_dist)
        for x in self._xticks:
            pygame.draw.line(
                self._screen,
                BLACK,
                start_pos = [coordinate(np.array([x, 0]))[0], screen_origin[1] - 1],
                end_pos = [coordinate(np.array([x, 0]))[0], screen_origin[1] + 1],
                width=1,
            )
            label = my_font.render(f"{x:.1f}", True, GREY)
            self._screen.blit(label, (coordinate(np.array([x, 0]))[0], screen_origin[1] + 2))
        # y-axis
        pygame.draw.line(
            self._screen,
            GREY,
            start_pos = [screen_origin[0], 0],
            end_pos=[screen_origin[0], self.screen_height],
            width=2,
        )
        # y-ticks
        scale = int(np.floor(np.log10(y_range/n_ticks)))   # potency of 10
        y_tick_dist = max(10 ** scale, y_range / 20)
        if self._yticks is None or self._yticks[0] > min_pos_y or self._yticks[-1] < max_pos_y:
            # Recomputation only if needed, make yticks larger then rendering --> smoother animation
            min_tick = np.round(min_pos_y - 5 * self._border_padding - y_tick_dist, -scale)
            max_tick = np.round(max_pos_y + 5 * self._border_padding + y_tick_dist, -scale)
            self._yticks = np.arange(min_tick, max_tick, y_tick_dist)
        for y in self._yticks:
            pygame.draw.line(
                self._screen,
                BLACK,
                start_pos = [screen_origin[0] - 1, coordinate(np.array([0, y]))[1]],
                end_pos = [screen_origin[0] + 1, coordinate(np.array([0, y]))[1]],
                width=1,
            )
            label = my_font.render(f"{y:.1f}", True, GREY)
            self._screen.blit(label, (screen_origin[0] + 2, coordinate(np.array([0, y]))[1]))


        # Environment objects
        goal = coordinate(self._task['goal'])
        pygame.draw.rect(# Goal
            self._screen,
            GREEN,
            [goal[0]-TASK_SIZE/2, goal[1]-TASK_SIZE/2, TASK_SIZE, TASK_SIZE]
        )
        agent = coordinate(self._last_state[:2])
        pygame.draw.rect(# Agent
            self._screen,
            BLACK,
            [agent[0]-AGENT_SIZE/2, agent[1]-AGENT_SIZE/2, AGENT_SIZE, AGENT_SIZE]
        )
        pygame.draw.line(# Action
            self._screen,
            ORANGE,
            start_pos = [*agent],
            end_pos = [
                agent[0] + ACTION_FACTOR * self._last_action[0] / x_tick_dist, 
                agent[1] - ACTION_FACTOR * self._last_action[1] / y_tick_dist
            ],
            width=8,
        )
        pygame.draw.line(# Clipped action
            self._screen,
            RED,
            start_pos = [agent[0], agent[1]],
            end_pos = [
                agent[0] + ACTION_FACTOR * self._last_clipped_action[0] / x_tick_dist, 
                agent[1] - ACTION_FACTOR * self._last_clipped_action[1] / y_tick_dist
            ],
            width=8,
        )


class Toy2dContinuous(Toy2D):
    """Toy environment with discretized continuous dynamics of a Mass-Spring-Damper system:
        x_1: Position in x-direction, x_2: Position in y-direction \n
        x_3: Velocity in x-direction, x_4: Velocity in y-direction \n
        x_1' = x_3 \n
        x_2' = x_4 \n
        x_3' = - k/m * x_1 - d/m * x_3 + 1/m * u_x \n
        x_4' = - k/m * x_2 - d/m * x_4 + 1/m * u_y \n
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
        ``'fixed'``: Goals are distributed linearly between min_pos and max_pos,
                    if min_pos or max_pos are infinite, tasks are placed between
                    -task_scale and +task_scale
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
    mass : float, optional
        Mass m, by default 1.0
    spring_constant : float, optional
        Spring constant k, by default 0.0
    damping : float, optional
        Damping factor d, by default 0.0
    min_pos : float, optional
        Left and bottom border of the environment, by default -1.0
    max_pos : float, optional
        Right and top border of the environment, by default 1.0
    max_action : float, optional
        Maximum norm of the action, by default 0.1
    max_vel : float, optional
        Maximum (norm) velocity value, by default 1.0
    """
    render_fps = 60
    def __init__(
            self,
            n_train_tasks: int,
            n_eval_tasks: int,
            task_generation_mode: str = 'random',
            one_sided_tasks: bool = False,
            task_scale: float = 1.0,
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
            n_train_tasks=n_train_tasks, n_eval_tasks=n_eval_tasks, 
            task_generation_mode=task_generation_mode, 
            one_sided_tasks=one_sided_tasks,
            task_scale=task_scale,
            change_steps=change_steps, change_prob=change_prob, 
            min_pos=min_pos, max_pos=max_pos, max_action=max_action,
            *args, **kwargs
        )
        self.max_vel = max_vel

        self.action_space = gym.spaces.Box(
            low=np.array([-max_vel, -max_vel]), 
            high=np.array([max_vel, max_vel]),
            shape=(2,),
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([self.min_pos_x, self.min_pos_y, -self.max_vel, -self.max_vel]), 
            high=np.array([self.max_pos_x, self.max_pos_y, self.max_vel, self.max_vel]),
            shape=(4,),
        )

        # Continuous dynamics matrices: x' = Ax + Bu where x is the state vector and u is the input
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-spring_constant/mass, 0, -damping/mass, 0],
            [0, -spring_constant/mass, 0, -damping/mass]
        ])
        B = np.array([[0, 0], [0, 0], [1/mass, 0], [0, 1/mass]])

        # Discretized dynamics matrices
        # A_d = e^{At}, B_d = A^{-1} (A_d - I) B
        # Discrete dynamics: x[k+1] = A_d x[k] + B_d u[k] where x is the state vector and u is the input
        self.A = expm(A * DELTA_T)
        if np.linalg.det(A) == 0:
            A_inv = np.linalg.pinv(A)
            self.B = A_inv @ (self.A - np.eye(*self.A.shape)) @ B
        else: 
            self.B = np.linalg.solve(A, (self.A - np.eye(*self.A.shape)) @ B)

        self._x_border_reached = False
        self._y_border_reached = False

        # Pygame rendering ...
        self._border_padding = max_vel if max_vel < np.inf else 100.0        # Padding for rendering of infinite environments

    def reset(self) -> Any:
        _, info = super().reset()
        self.state[2:] = 0
        return self.state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self._try_task_update()

        # Limit action (force)
        force = action * np.min([1.0, self.max_action / np.linalg.norm(action)])
        assert self.action_space.contains(force.astype(np.float32)), "Clipped action is not in action space!"

        # Apply dynamics
        self._last_state = self.state
        self.state = self.A @ self.state + self.B @ force

        # Limit states
        self.state = np.clip(
            self.state, 
            a_min=[self.min_pos_x, self.min_pos_y, -self.max_vel, -self.max_vel], 
            a_max=[self.max_pos_x, self.max_pos_y, self.max_vel, self.max_vel]
        )

        # Limit velocity to max_vel
        vel = np.array([self.state[2], self.state[3]])
        self.state[2] *= np.min([1.0, self.max_vel / np.linalg.norm(vel)])
        self.state[3] *= np.min([1.0, self.max_vel / np.linalg.norm(vel)])

        # Set velocity to zero if border is reached
        if (self.state[0] == self.min_pos_x or self.state[0] == self.max_pos_x) and not self._x_border_reached:
            self.state[2] = 0
            self._x_border_reached = True
        else:
            self._x_border_reached = False
        if (self.state[1] == self.min_pos_y or self.state[1] == self.max_pos_y) and not self._y_border_reached:
            self.state[3] = 0
            self._y_border_reached = True
        else:
            self._y_border_reached = False
        assert self.observation_space.contains(self.state.astype(np.float32)), "State is not in state space!"

        # Reward
        if self._reward_type == "L1":
            reward = - np.linalg.norm(self.state[:2] - self._task['goal'], ord=1)
        elif self._reward_type == "L2":
            reward = - np.linalg.norm(self.state[:2] - self._task['goal'], ord=2)
        else:
            raise ValueError(f"Unknown option for reward type: {self._reward_type}")

        done = False
        env_info = {
            'true_task': self._task
        }

        self._steps_since_task_update += 1
        self._last_action = action
        self._last_clipped_action = force

        return self.state, reward, done, False, env_info
