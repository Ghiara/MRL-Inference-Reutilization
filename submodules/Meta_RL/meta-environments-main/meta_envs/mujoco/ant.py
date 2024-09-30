"""
This module contains derived versions of the AntEnv which implement
the TaskSetEnvironment interface.

The classes in this module use pygame for rendering!

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-07
"""

from gym.envs.mujoco.ant import AntEnv
from gym.utils.ezpickle import EzPickle
from typing import List, Tuple
import numpy as np
import mujoco_py

from ..base import TaskSetEnvironment, Task
from ..pygame_rendering import PygameRenderer


class Ant(AntEnv):
    """
    A subclass of the standard ant environment which uses pygame rendering
    with a custom camera.
    """
    def __init__(self, render_mode: str = 'human', **kwargs):
        super().__init__(render_mode=render_mode, **kwargs)
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
        self.renderer.render_image(img, title="Ant")
        self.renderer.clock.tick(self.metadata['render_fps'])
        
    def _initialize_camera(self):
        # set camera parameters for viewing
        sim = self.sim
        self.viewer = mujoco_py.MjRenderContextOffscreen(sim)
        camera = self.viewer.cam
        camera.type = 1
        camera.trackbodyid = 0
        camera.distance += 5
        camera.elevation = -70
        sim.add_render_context(self.viewer)


class AntGoal(Ant, TaskSetEnvironment, EzPickle):
    """An ant meta-environment which uses goal positions on the x-y-plane
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
        ``'random'``: Goals are sampled from a Gaussian with standard deviation 10.0
        ``'fixed'``: Goals are placed equally distanced between -10.0 and 10.0
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
        Ant.__init__(self, render_mode=render_mode)
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
        
        pos = np.array(self.get_body_com("torso"))
        obs, rew, term, truncated, info = super().step(action)
        info['true_task'] = self.task
        rew = - np.linalg.norm(pos[:2] - self.task['goal'])
        return obs, rew.item(), term, truncated, info
    
    def _init_tasks(
        self, 
        n_train_tasks: int, 
        n_eval_tasks: int, 
        mode: str = 'random', 
        one_sided: bool = False
    ) -> Tuple[List[Task], List[Task]]:
        std = 10
        if mode == 'random':
            train_goals = std * np.random.randn(n_train_tasks, 2)
            test_goals = std * np.random.randn(n_eval_tasks, 2)
        elif mode == 'fixed':
            grid_len = int(np.ceil(np.sqrt(n_train_tasks)))
            x = np.linspace(-std, std, grid_len)
            y = np.linspace(-std, std, grid_len)
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

    def render(self, mode: str = None, width: int = 800, height: int = 400):
        del self.viewer._markers[:]
        self.viewer.add_marker(
            pos=np.array([*self.task['goal'], 0]),
            label="goal"
        )
        return super().render(mode, width, height)


class AntVel(Ant, TaskSetEnvironment, EzPickle):
    """An ant meta-environment which uses target velocities in the x-y-plane
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
        Ant.__init__(self, render_mode=render_mode)
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
        
        pos_before = np.array(self.get_body_com("torso"))
        obs, rew, done, trunc, info = super().step(action)
        pos_after = np.array(self.get_body_com("torso"))
        velocity = (pos_after - pos_before) / self.dt

        rew = -1.0 * np.linalg.norm(velocity[:2] - self._task['velocity'])
        info['true_task'] = self._task
        info['veloctiy'] = velocity
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
            train_goals = std * np.random.randn(n_train_tasks, 2)
            test_goals = std * np.random.randn(n_eval_tasks, 2)
        elif mode == 'fixed':
            grid_len = int(np.ceil(np.sqrt(n_train_tasks)))
            x = np.linspace(-std, std, grid_len)
            y = np.linspace(-std, std, grid_len)
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
            train_tasks.append({'id': next_id, 'velocity': np.array([goal])})
            next_id += 1
        for goal in test_goals:
            eval_tasks.append({'id': next_id, 'velocity': np.array([goal])})
            next_id += 1
        return train_tasks, eval_tasks


    