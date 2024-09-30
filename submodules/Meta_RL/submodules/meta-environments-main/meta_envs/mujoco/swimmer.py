import gym
from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from gym.utils.ezpickle import EzPickle
from typing import List, Tuple
import numpy as np
import mujoco_py
from ..pygame_rendering import PygameRenderer

class SwimmerGoal(SwimmerEnv):
    # rewrite this to give as input the dictionary
    def __init__(self, healthy_scale = 1, render_mode: str = 'rgb_array', treminal_threshold = 0.1, *args, **kwargs):
        super().__init__(render_mode=render_mode, exclude_current_positions_from_observation=False,*args, **kwargs)
        self.healthy_scale = healthy_scale
        self.screen_height = 400
        self.screen_width = 400
        self.task = 0 
        self.terminal_threshold = treminal_threshold

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

        # How to make the reward robust (combination of healthy reward, distance_to_goal)
    def step(self, action):
        # previous_pos = self.get_body_com("torso")
        obs, rew, term, truncated, info = super().step(action)
        rew = - np.abs(obs[0] - self.task)

        if self.terminal_threshold is not None:
            if abs(rew) < self.terminal_threshold:
                term = 1
        info['reward_goal'] = rew.item()
        rew += info['reward_ctrl']
        # rew += info['reward_ctrl']
        
        info['true_task'] = self.task
        return obs, rew.item(), term, truncated, info
    def update_task(self, task):
        self.task = task

    # def reset_model(self):
    #     qpos = self.init_qpos
    #     qvel = self.init_qvel
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()
