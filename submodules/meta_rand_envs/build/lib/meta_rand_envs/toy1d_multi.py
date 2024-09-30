import numpy as np
import pygame
import gym
from scipy.linalg import expm
from typing import Dict, Tuple, Any, List
import random
import torch

from meta_envs.toy_goal.base import ToyEnv, Task

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


class Toy1dMultiTask(ToyEnv):
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
        

        '''''''''

        ### From Halfcheetahmulti

        '''''''''
        self.meta_mode = 'train'
        self.change_mode = kwargs.get('change_mode', '')
        self.change_prob = kwargs.get('change_prob', 1.0)
        self.change_steps = kwargs.get('change_steps', 80)
        self.termination_possible = kwargs.get('termination_possible', False)
        self.steps = 0

        # velocity/position specifications in ranges [from, to]
        self.velocity_x = [1.0, 5.0]
        self.pos_x = [1.0, 15.0]
        self.velocity_y = [2. * np.pi, 4. * np.pi]
        self.pos_y = [np.pi / 6., np.pi / 2.]
        self.velocity_z = [1.5, 3.]
        self.dt = 1

        # self.velocity_x = [1.0, 4.0]
        # self.pos_x = [5.0, 10.0]
        # self.velocity_y = [2. * np.pi, 4. * np.pi]
        # self.pos_y = [np.pi / 5., np.pi / 2.]
        # self.velocity_z = [1.5, 3.]

        self.positive_change_point_basis = kwargs.get('positive_change_point_basis', 10)
        self.negative_change_point_basis = kwargs.get('negative_change_point_basis', -10)
        self.change_point_interval = kwargs.get('change_point_interval', 1)
        self.base_task = 0
        self.task_specification = 1.0
        task_names = ['velocity_forward', 'velocity_backward',
                      'goal_forward', 'goal_backward',
                      'flip_forward',
                      'stand_front', 'stand_back',
                      'jump',
                      'direction_forward', 'direction_backward',
                      'velocity']
        self.task_variants = kwargs.get('task_variants', task_names)
        self.bt2t = {k: self.task_variants.index(k) if k in self.task_variants else -1 for k in task_names}

        self.positive_change_point = self.positive_change_point_basis + np.random.random() * self.change_point_interval
        self.negative_change_point = self.negative_change_point_basis - np.random.random() * self.change_point_interval


        '''''''''

        ### From Toy1d

        '''''''''
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.max_action = max_action

        self.observation_space = gym.spaces.Box(low=self.min_pos, high=self.max_pos, shape=(3,))
        self.action_space = gym.spaces.Box(low=-self.max_action, high=self.max_action, shape=(3,))
        self.state = np.zeros((3))

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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if isinstance(action, torch.Tensor):
            clipped_action = action.clip(-self.max_action, self.max_action).cpu().numpy()
            clipped_action = np.array([clipped_action]) 
        else:
            clipped_action = action.clip(-self.max_action, self.max_action)
        assert self.action_space.contains(clipped_action.astype(np.float32)), "Clipped action is not in action space!"

        self._last_state = self.state
        self.state = (self.state + clipped_action).clip(self.min_pos, self.max_pos)
        xposbefore = self._last_state
        xvelafter = (self.state - self._last_state)/self.dt 
        xposafter = self.state
        assert self.observation_space.contains(self.state.astype(np.float32)), "State is not in state space!"
        reward = - float(np.abs(self.state - self._task['goal']))
        done = False
        env_info = {
            'true_task': self._task
        }

        self._steps_since_task_update += 1

        self._last_action = action.item()
        self._last_clipped_action = clipped_action.item()

        if self.base_task in [self.bt2t['velocity_forward'], self.bt2t['velocity_backward']]:  # 'velocity'
            reward_run = - np.abs(xvelafter[0] - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)

        elif self.base_task in [self.bt2t['goal_forward'], self.bt2t['goal_backward']]:  # 'goal'
            reward_run = - np.abs(xposafter[0] - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)

        elif self.base_task in [self.bt2t['flip_forward']]:  # 'flipping'
            reward_run = - np.abs(xvelafter[2] - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)

        elif self.base_task in [self.bt2t['stand_front'], self.bt2t['stand_back']]:  # 'stand_up'
            reward_run = - np.abs(xposafter[2] - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)

        elif self.base_task in [self.bt2t['jump']]:  # 'jump'
            reward_run = - np.abs(np.abs(xvelafter[1]) - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)

        elif self.base_task in [self.bt2t['direction_forward'], self.bt2t['direction_backward']]:  # 'direction'
            reward_run = (xposafter[0] - xposbefore[0]) / self.dt * self.task_specification
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run

        elif self.base_task in [self.bt2t['velocity']]:
            forward_vel = (xposafter[0] - xposbefore[0]) / self.dt
            reward_run = -1.0 * np.abs(forward_vel - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run
        else:
            raise RuntimeError("base task not recognized")

        # print(str(self.base_task) + ": " + str(reward))
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        # if self.termination_possible:
        #     state = self.state_vector()
        #     notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
        #     done = not notdone
        # else:
        #     done = False
        self.steps += 1
        return self.state, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
                                      true_task=dict(base_task=self.base_task, specification=self.task_specification))

        return self.state, reward, done, False, env_info

    def reset(self) -> Tuple[np.ndarray, Dict]:
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
  
    def sample_tasks(self, num_tasks_list):
        if type(num_tasks_list) != list: num_tasks_list = [num_tasks_list]

        num_base_tasks = len(self.task_variants)
        num_tasks_per_subtask = [int(num_tasks / num_base_tasks) for num_tasks in num_tasks_list]
        num_tasks_per_subtask_cumsum = np.cumsum(num_tasks_per_subtask)

        tasks = [[] for _ in range(len(num_tasks_list))]
        # velocity tasks
        if 'velocity_forward' in self.task_variants:
            velocities = np.linspace(self.velocity_x[0], self.velocity_x[1], num=sum(num_tasks_per_subtask))
            tasks_velocity = [
                {'base_task': self.bt2t['velocity_forward'], 'specification': velocity, 'color': np.array([1, 0, 0])}
                for velocity in velocities]
            np.random.shuffle(tasks_velocity)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_velocity[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        if 'velocity_backward' in self.task_variants:
            velocities = np.linspace(-self.velocity_x[1], -self.velocity_x[0], num=sum(num_tasks_per_subtask))
            tasks_velocity = [
                {'base_task': self.bt2t['velocity_backward'], 'specification': velocity, 'color': np.array([0, 1, 0])}
                for velocity in velocities]
            np.random.shuffle(tasks_velocity)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_velocity[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # goal
        if 'goal_forward' in self.task_variants:
            goals = np.linspace(self.pos_x[0], self.pos_x[1], num=sum(num_tasks_per_subtask))
            tasks_goal = [{'base_task': self.bt2t['goal_forward'], 'specification': goal, 'color': np.array([1, 1, 0])}
                          for goal in goals]
            np.random.shuffle(tasks_goal)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_goal[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        if 'goal_backward' in self.task_variants:
            goals = np.linspace(-self.pos_x[1], -self.pos_x[0], num=sum(num_tasks_per_subtask))
            tasks_goal = [{'base_task': self.bt2t['goal_backward'], 'specification': goal, 'color': np.array([0, 1, 1])}
                          for goal in goals]
            np.random.shuffle(tasks_goal)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_goal[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # flipping
        if 'flip_forward' in self.task_variants:
            goals = np.linspace(self.velocity_y[0], self.velocity_y[1], num=sum(num_tasks_per_subtask))
            tasks_flipping = [
                {'base_task': self.bt2t['flip_forward'], 'specification': goal, 'color': np.array([0.5, 0.5, 0])} for
                goal in goals]
            np.random.shuffle(tasks_flipping)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_flipping[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # stand_up
        if 'stand_front' in self.task_variants:
            goals = np.linspace(self.pos_y[0], self.pos_y[1], num=sum(num_tasks_per_subtask))
            tasks_stand_up = [
                {'base_task': self.bt2t['stand_front'], 'specification': goal, 'color': np.array([1., 0, 0.5])} for goal
                in goals]
            np.random.shuffle(tasks_stand_up)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_stand_up[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        if 'stand_back' in self.task_variants:
            goals = np.linspace(-self.pos_y[1], -self.pos_y[0], num=sum(num_tasks_per_subtask))
            tasks_stand_up = [
                {'base_task': self.bt2t['stand_back'], 'specification': goal, 'color': np.array([0.5, 0, 1.])} for goal
                in goals]
            np.random.shuffle(tasks_stand_up)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_stand_up[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # jump
        if 'jump' in self.task_variants:
            goals = np.linspace(self.velocity_z[0], self.velocity_z[1], num=sum(num_tasks_per_subtask))
            tasks_jump = [{'base_task': self.bt2t['jump'], 'specification': goal, 'color': np.array([0.5, 0.5, 0.5])}
                          for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # direction
        if 'direction_forward' in self.task_variants:
            goals = np.array([1.] * sum(num_tasks_per_subtask))
            tasks_jump = [
                {'base_task': self.bt2t['direction_forward'], 'specification': goal, 'color': np.array([0.5, 0.5, 0.])}
                for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]
        if 'direction_backward' in self.task_variants:
            goals = np.array([-1.] * sum(num_tasks_per_subtask))
            tasks_jump = [
                {'base_task': self.bt2t['direction_backward'], 'specification': goal, 'color': np.array([0.5, 0., 0.5])}
                for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]
        if 'velocity' in self.task_variants:
            goals = np.linspace(0.0, 3.0, num=sum(num_tasks_per_subtask))
            tasks_jump = [{'base_task': self.bt2t['velocity'], 'specification': goal, 'color': np.array([0.5, 0., 0.5])}
                          for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # Return nested list only if list is given as input
        return tasks if len(num_tasks_list) > 1 else tasks[0]




    def viewer_setup(self):
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = 0

    def change_task(self, spec):
        self.base_task = spec['base_task']
        self.task_specification = spec['specification']
        self._goal = spec['specification']
        self.color = spec['color']
        self.recolor()

    def recolor(self):
        # geom_rgba = self._init_geom_rgba.copy()
        # rgb_value = self.color
        # geom_rgba[1:, :3] = np.asarray(rgb_value)
        # self.model.geom_rgba[:] = geom_rgba
        pass

    def set_meta_mode(self, mode):
        self.meta_mode = mode


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
    
    def reset(self) -> Any:
        _, info = super().reset()
        self.state[1] = 0
        return self.state, info
    
    def _get_obs(self):
        return self.state

    def _draw_env(self):

        # Determine rendering boundaries
        min_pos, max_pos = self.pos_x[0], self.pos_x[1]
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
            coordinate(float(self._last_state[1])) - AGENT_SIZE/2, AGENT_SIZE, AGENT_SIZE]
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
