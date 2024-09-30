import gym
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
from gym.utils.ezpickle import EzPickle
from typing import List, Tuple
import numpy as np
import mujoco_py
from ..pygame_rendering import PygameRenderer

class WalkerGoal(Walker2dEnv):
    # rewrite this to give as input the dictionary
    def __init__(self, healthy_scale = 1, render_mode: str = 'rgb_array', *args, **kwargs):
        super().__init__(render_mode=render_mode, exclude_current_positions_from_observation=False,*args, **kwargs)
        self.healthy_scale = healthy_scale
        self.screen_height = 400
        self.screen_width = 400
        self.task = 0 

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
        distance_before = np.abs(self.task - self.get_body_com('torso')[0])
        state, reward, done, _, info = super().step(action)
        distance_after = np.abs(self.task - self.get_body_com('torso')[0])
        ctrl_cost = self.control_cost(action)
        healthy_penalty = 0

        # Compute the new reward based on the distance to the goal
        # Assuming the horizontal position of the Hopper is given by the x-coordinate (usually the first state variable)
        # distance_to_goal = np.abs(self.get_body_com('foot')[0] - self.task)
        step_reward = -(distance_after - distance_before)/self.dt
        # reward = - distance_after
        healthy_reward = self._healthy_reward
        if not self.is_healthy:
            healthy_penalty = -10

        rewards = step_reward + healthy_reward * self.healthy_scale + healthy_penalty# Reward is negative distance to the goal
        reward = rewards - ctrl_cost
        terminated = self.terminated

        return state, reward, terminated, False, info

    def update_task(self, task):
        self.task = task

    def reset(self):
        obs = super().reset()
        new_obs = np.append(self.get_body_com("foot")[0], obs[0])
        return new_obs, {}

    # def reset_model(self):
    #     qpos = self.init_qpos
    #     qvel = self.init_qvel
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()
