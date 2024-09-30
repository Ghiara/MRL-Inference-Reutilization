"""
This module contains derived versions of the HalfCheetahEnv which implement
the TaskSetEnvironment interface.

The classes in this module use pygame for rendering!

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-22
"""

import gym
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv as HalfCheetahEnv_
from gym.utils.ezpickle import EzPickle
from typing import List, Tuple
import numpy as np
import mujoco_py

from ..base import TaskSetEnvironment, Task, MultiTaskEnv
from ..pygame_rendering import PygameRenderer

class HalfCheetahEnv(HalfCheetahEnv_):
    """
    A subclass of the standard cheetah environment which 
        - uses pygame rendering with a custom camera
        - extends the observations by the position of the cheetah's torso (first
            three dimensions of the observations)

    The observation space of this environment has shape (20, ).
    """
    def __init__(self, render_mode: str = 'rgb_array', *args, **kwargs):
        super().__init__(render_mode=render_mode, *args, **kwargs)

        # Override observation space of standard cheetah
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)

        # Rendering
        self.renderer = None
        self.viewer = None
        if self.render_mode == 'human':
            self.renderer = PygameRenderer()
        self._initialize_camera()

    def render(self, mode: str = None, width: int = 800, height: int = 400):
        if mode is None: mode = self.render_mode

        img = self.sim.render(
            width=width,
            height=height,
            camera_name=None,
        )
        img = np.rot90(img, 3)

        if mode != 'human': 
            return img
        if self.renderer is None:
            self.renderer = PygameRenderer()
        self.renderer.render_image(img, title="Half-Cheetah")
        self.renderer.clock.tick(self.metadata['render_fps'])
        
    def _initialize_camera(self):
        # set camera parameters for viewing
        sim = self.sim
        self.viewer = mujoco_py.MjRenderContextOffscreen(sim)
        camera = self.viewer.cam
        camera.type = 1
        camera.trackbodyid = 0
        camera.elevation = -20
        sim.add_render_context(self.viewer)

    def _get_obs(self):
        obs = super()._get_obs()
        obs = np.concatenate([self.get_body_com('torso'), obs], axis=-1)
        return obs


class HalfCheetahDir(HalfCheetahEnv, TaskSetEnvironment, EzPickle):
    """A half-cheetah meta-environment which uses x-direction velocity for reward.

    Parameters
    ----------
    one_sided_tasks : bool, optional
        Set to True to map all goal positions to the positive axis.
        By default False
    change_steps : int, optional
        Number of steps until which the task can change, by default 100
    change_prob : float, optional
        Probability of a task change (after ``change_steps``), by default 1.0
    render_mode : str, optional
        Determines the render mode of the environment. Can be one of 
        | ``'human'`` | ``'rgb_array'`` |
        By default 'human'
    """

    def __init__(
        self,
        one_sided_tasks: bool = False,
        change_steps: int = 100,
        change_prob: float = 1.0,
        render_mode : str = 'human',
        *args,
        **kwargs,
    ) -> None:
        train_tasks, eval_tasks = self._init_tasks(one_sided=one_sided_tasks)
        HalfCheetahEnv.__init__(self, render_mode=render_mode)
        TaskSetEnvironment.__init__(
            self, 
            train_tasks, eval_tasks, 
            change_steps=change_steps, change_prob=change_prob, 
            *args, **kwargs
        )
        EzPickle.__init__(  # Required to support copy.deepcopy()
            self,
            one_sided_tasks=one_sided_tasks,
            change_steps=change_steps,
            change_prob=change_prob,
            render_mode=render_mode,
            *args,
            **kwargs,
        )

    def step(self, action):
        self._try_task_update()
        self._steps_since_task_update += 1

        xposbefore = self.get_body_com("torso")[0]
        obs, rew, done, trunc, info = super().step(action)
        xposafter = self.get_body_com("torso")[0]
        velocity = (xposafter - xposbefore)/self.dt

        rew = self._task['direction'] * info['reward_run']
        rew += info['reward_ctrl']
        info['true_task'] = self._task
        info['velocity'] = velocity
        info['reward_run'] = info['reward_run'] * self._task['direction']
        return (obs, rew.item(), done, trunc, info)
    
    def _init_tasks(
        self, 
        one_sided: bool = False
    ) -> Tuple[List[Task], List[Task]]:

        tasks = [
            {'id': 0, 'direction': np.array([1.0])},
            {'id': 1, 'direction': np.array([-1.0])},
        ]
        if one_sided:
            tasks = tasks[:1]
        
        return tasks, tasks

    def render(self, mode: str = None, width: int = 800, height: int = 400):
        del self.viewer._markers[:]
        self.viewer.add_marker(
            pos=np.array([3.0 * float(self.task['direction']) + self.get_body_com('torso')[0], 0, 0]),
            label="direction"
        )
        return super().render(mode, width, height)


class HalfCheetahDirOpenTask(HalfCheetahDir):
    @property
    def task_encoding_shape(self) -> Tuple[int, ...]:
        return (1, )
    
    def step(self, action) -> Tuple[np.ndarray, np.ndarray, float, bool, bool, dict]:
        obs, rew, term, trunc, info = super().step(action)
        return obs, self.task['direction'], rew, term, trunc, info

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        return obs, self.task['direction'], info


class HalfCheetahGoal(HalfCheetahEnv, TaskSetEnvironment, EzPickle):
    """A half-cheetah meta-environment which uses goal positions on the x-axis
    to determine the reward.

    Goal positions are sampled from a Gaussian distribution with mean zero and
    standard deviation ``task_scale`` (if ``task_generation_mode = 'random'``),
    or distributed linearly between ``-task_scale`` and ``task_scale`` (if 
    ``task_generation_mode = 'fixed'``).
    In case ``one_sided_tasks = True``, the tasks will be mapped to positive
    goal positions by taking the absolute value.

    NOTE: The original cheetah observations seem to lack any information about
    the absolute position of the agent in cartesian coordinates.
    This environment model adds the cartesian coordinates of the cheetah's torso
    to the beginning of the observations.
    This results in a 20-dimensional observation space.

    Parameters
    ----------
    n_train_tasks : int
        Number of train tasks.
    n_eval_tasks : int
        Number of test tasks.
    task_generation_mode : str, optional
        Determines how tasks are generated:
        | ``'random'`` | ``'fixed'`` |, by default 'random'
        ``'random'``: Goals are sampled from a Gaussian with standard deviation 10.0
        ``'fixed'``: Goals are placed equally distanced between -10.0 and 10.0
    one_sided_tasks : bool, optional
        Set to True to map all goal positions to the positive axis.
        By default False
    task_scale : float, optional
        Range for goal positions (see above for detailed explanation).
        By default 25.0
    change_steps : int, optional
        Number of steps until which the task can change, by default 100
    change_prob : float, optional
        Probability of a task change (after ``change_steps``), by default 1.0
    render_mode : str, optional
        Determines the render mode of the environment. Can be one of 
        | ``'human'`` | ``'rgb_array'`` |
        By default 'human'
    """

    def __init__(
        self,
        n_train_tasks: int,
        n_eval_tasks: int,
        task_generation_mode: str = 'random',
        one_sided_tasks: bool = False,
        task_scale: float = 25.0,
        change_steps: int = 100,
        change_prob: float = 1.0,
        render_mode : str = 'human',
        *args,
        **kwargs,
    ) -> None:
        train_tasks, eval_tasks = self._init_tasks(
            n_train_tasks, n_eval_tasks, 
            mode=task_generation_mode, one_sided=one_sided_tasks
        )
        HalfCheetahEnv.__init__(self, render_mode=render_mode)
        TaskSetEnvironment.__init__(
            self, 
            train_tasks, eval_tasks, 
            change_steps=change_steps, change_prob=change_prob, 
            task_scale=task_scale, 
            *args, **kwargs
        )
        EzPickle.__init__(  # Required to support copy.deepcopy()
            self,
            n_train_tasks=n_train_tasks,
            n_eval_tasks=n_eval_tasks,
            task_generation_mode=task_generation_mode,
            one_sided_tasks=one_sided_tasks,
            task_scale=task_scale,
            change_steps=change_steps,
            change_prob=change_prob,
            render_mode=render_mode,
            *args,
            **kwargs,
        )

    def step(self, action):
        self._try_task_update()
        self._steps_since_task_update += 1

        obs, rew, term, truncated, info = super().step(action)
        pos = np.array(self.get_body_com("torso"))

        rew = - np.abs(pos[0] - self.task['goal'])
        info['reward_goal'] = rew.item()
        rew += info['reward_ctrl']

        info['true_task'] = self.task
        return obs, rew.item(), term, truncated, info
    
    def _init_tasks(
        self, 
        n_train_tasks: int, 
        n_eval_tasks: int, 
        mode: str = 'random', 
        one_sided: bool = False,
        task_scale: float = 25.0,
    ) -> Tuple[List[Task], List[Task]]:
        if mode == 'random':
            train_goals = task_scale * np.random.randn(n_train_tasks)
            test_goals = task_scale * np.random.randn(n_eval_tasks)
        elif mode == 'fixed':
            train_goals = np.linspace(-task_scale, task_scale, n_train_tasks)
            test_goals = np.linspace(-task_scale, task_scale, n_eval_tasks)
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

    def render(self, mode: str = None, width: int = 800, height: int = 400):
        del self.viewer._markers[:]
        self.viewer.add_marker(
            pos=np.array([*self.task['goal'].flatten(), 0, 0]),
            label="goal"
        )
        return super().render(mode, width, height)


class HalfCheetahGoalOpenTask(HalfCheetahGoal, MultiTaskEnv):

    @property
    def task_encoding_shape(self) -> Tuple[int, ...]:
        return (1, )
    
    def step(self, action) -> Tuple[np.ndarray, np.ndarray, float, bool, bool, dict]:
        obs, rew, term, trunc, info = super().step(action)
        return obs, self.task['goal'], rew, term, trunc, info

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        return obs, self.task['goal'], info


class HalfCheetahVel(HalfCheetahEnv, TaskSetEnvironment, EzPickle):
    """A half-cheetah meta-environment which uses target velocities in x-direction
    to determine the reward.

    Parameters
    ----------
    n_train_tasks : int
        Number of train tasks.
    n_eval_tasks : int
        Number of test tasks.
    task_generation_mode : str, optional
        Determines how tasks are generated:
        | ``'random'`` | ``'fixed'`` |, by default 'random'
        ``'random'``: Velocities are sampled from a Gaussian with standard deviation 10.0
        ``'fixed'``: Velocities are generated equally spreaded between -10.0 and 10.0
    one_sided_tasks : bool, optional
        Set to True to map all goal positions to the positive axis.
        By default False
    change_steps : int, optional
        Number of steps until which the task can change, by default 100
    change_prob : float, optional
        Probability of a task change (after ``change_steps``), by default 1.0
    render_mode : str, optional
        Determines the render mode of the environment. Can be one of 
        | ``'human'`` | ``'rgb_array'`` |
        By default 'human'
    """

    def __init__(
        self,
        n_train_tasks: int,
        n_eval_tasks: int,
        task_generation_mode: str = 'random',
        one_sided_tasks: bool = False,
        change_steps: int = 100,
        change_prob: float = 1.0,
        render_mode: str = 'human',
        *args,
        **kwargs,
    ) -> None:
        train_tasks, eval_tasks = self._init_tasks(
            n_train_tasks, n_eval_tasks, 
            mode=task_generation_mode, one_sided=one_sided_tasks
        )
        HalfCheetahEnv.__init__(self, render_mode=render_mode)
        TaskSetEnvironment.__init__(
            self, 
            train_tasks, eval_tasks, 
            change_steps=change_steps, change_prob=change_prob, 
            *args, **kwargs
        )
        EzPickle.__init__(  # Required to support copy.deepcopy()
            self,
            n_train_tasks=n_train_tasks,
            n_eval_tasks=n_eval_tasks,
            task_generation_mode=task_generation_mode,
            one_sided_tasks=one_sided_tasks,
            change_steps=change_steps,
            change_prob=change_prob,
            render_mode=render_mode,
            *args,
            **kwargs,
        )

    def step(self, action):
        self._try_task_update()
        self._steps_since_task_update += 1

        xposbefore = self.get_body_com("torso")[0]
        obs, rew, done, trunc, info = super().step(action)
        xposafter = self.get_body_com("torso")[0]
        velocity = (xposafter - xposbefore)/self.dt

        rew = -1.0 * abs(velocity - self._task['velocity'])
        info['reward_velocity'] = rew.item()
        rew += info['reward_ctrl']

        info['true_task'] = self._task
        info['velocity'] = velocity
        return (obs, rew.item(), done, trunc, info)
    
    def _init_tasks(
        self, 
        n_train_tasks: int, 
        n_eval_tasks: int, 
        mode: str = 'random', 
        one_sided: bool = False
    ) -> Tuple[List[Task], List[Task]]:
        std = 10
        if mode == 'random':
            train_goals = std * np.random.randn(n_train_tasks)
            test_goals = std * np.random.randn(n_eval_tasks)
        elif mode == 'fixed':
            train_goals = np.linspace(-std, std, n_train_tasks)
            test_goals = np.linspace(-std, std, n_eval_tasks)
        else:
            raise ValueError(f'Unknown option \'{mode}\' for argument \'task_generation_mode\'.')
        
        if one_sided:
            train_goals = [np.abs(train_goal) for train_goal in train_goals]
            test_goals = [np.abs(test_goal) for test_goal in test_goals]
        
        train_tasks = []
        eval_tasks = []
        next_id = 0
        for goal in train_goals:
            train_tasks.append({'id': next_id, 'velocity': np.array([goal])})
            next_id += 1
        for goal in test_goals:
            eval_tasks.append({'id': next_id, 'velocity': np.array([goal])})
            next_id += 1
        return train_tasks, eval_tasks

class HalfCheetahVelOpenTask(HalfCheetahVel, MultiTaskEnv):

    @property
    def task_encoding_shape(self) -> Tuple[int, ...]:
        return (1, )
    
    def step(self, action) -> Tuple[np.ndarray, np.ndarray, float, bool, bool, dict]:
        obs, rew, term, trunc, info = super().step(action)
        return obs, self.task['velocity'], rew, term, trunc, info

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        return obs, self.task['velocity'], info
    

class HalfCheetahEnvExternalTask(HalfCheetahGoal):

    def __init__(
        self,
        n_train_tasks: int,
        n_eval_tasks: int,
        task_generation_mode: str = 'random',
        one_sided_tasks: bool = False,
        task_scale: float = 1.0,
        reward_scale:float = 1.0,
        change_steps: int = 1000000,
        change_prob: float = 0.0,
        render_mode : str = 'human',
        terminal_threshold : float = None,
        extended_reward=False,
        scale_angle_reward=1.0,
        *args,
        **kwargs,
    ) -> None:
        self.reward_scale = reward_scale
        self.extended_reward = extended_reward
        self.scale_angle_reward = scale_angle_reward
        self.terminal_threshold = terminal_threshold
        train_tasks, eval_tasks = self._init_tasks(
            n_train_tasks, n_eval_tasks, 
            mode=task_generation_mode, one_sided=one_sided_tasks
        )
        HalfCheetahEnv.__init__(self, render_mode=render_mode)
        TaskSetEnvironment.__init__(
            self, 
            train_tasks, eval_tasks, 
            change_steps=change_steps, change_prob=change_prob, 
            task_scale=task_scale, 
            *args, **kwargs
        )
        EzPickle.__init__(  # Required to support copy.deepcopy()
            self,
            n_train_tasks=n_train_tasks,
            n_eval_tasks=n_eval_tasks,
            task_generation_mode=task_generation_mode,
            one_sided_tasks=one_sided_tasks,
            task_scale=task_scale,
            change_steps=change_steps,
            change_prob=change_prob,
            render_mode=render_mode,
            *args,
            **kwargs,
        )
    def step(self, action):
        self._steps_since_task_update += 1
        # previous_pos = self.get_body_com("torso")
        obs, rew, term, truncated, info = super().step(action)
        pos = np.array(self.get_body_com("torso"))
        rew = - np.abs(pos[0] - self.task['goal']) * self.reward_scale
        if self.extended_reward:
            rew = - np.abs(pos[0] - self.task['goal']) * self.reward_scale - self.scale_angle_reward * np.abs(obs[4]-0.0)
        if self.terminal_threshold is not None:
            if abs(rew) < self.terminal_threshold:
                term = 1
        info['reward_goal'] = rew.item()
        rew += info['reward_ctrl']
        # rew += info['reward_ctrl']
        
        info['true_task'] = self.task
        return obs, rew.item(), term, truncated, info

    # def reset_model(self):
    #     qpos = self.init_qpos
    #     qvel = self.init_qvel
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()
    
    def update_task(self, task):
        self.task = dict(goal=task)
        # self.task['goal'] = self.task['goal'].flatten()