import gym
from gym.envs.mujoco.hopper import HopperEnv
from gym.utils.ezpickle import EzPickle
from typing import List, Tuple
import numpy as np
import mujoco_py
from meta_envs.pygame_rendering import PygameRenderer
from gym.spaces import Box


class HopperGoal(HopperEnv):
    # rewrite this to give as input the dictionary
    def __init__(self, healthy_scale = 1, render_mode: str = 'rgb_array', *args, **kwargs):
        super().__init__(render_mode=render_mode,*args, **kwargs)
        self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64
            )
        self.healthy_scale = healthy_scale
        self.screen_height = 400
        self.screen_width = 400
        self.task = 0

    # def render(self, mode: str = None, width: int = 800, height: int = 400):
    #     if mode is None: mode = self.render_mode

    #     self.sim.model.cam_pos[0][0] = self.get_body_com('torso')[0]
    #     self.sim.model.cam_pos0[0][0] = self.get_body_com('torso')[0]
    #     img = self.sim.render(
    #         width=width,
    #         height=height,
    #         camera_name=None,
    #     )
    #     img = np.rot90(img, 2)

    #     if mode != 'human': 
    #         return img
    #     if self.renderer is None:
    #         self.renderer = PygameRenderer()
    #     self.renderer.render_image(img, title="Half-Cheetah")
    #     self.renderer.clock.tick(self.metadata['render_fps'])
        
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
    def step(self, a):
        # posbefore = self.sim.data.qpos[0]
        # self.do_simulation(a, self.frame_skip)
        # posafter, height, ang = self.sim.data.qpos[0:3]

        # alive_bonus = 1.0
        # reward = (posafter - posbefore) / self.dt
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()
        # s = self.state_vector()
        # terminated = not (
        #     np.isfinite(s).all()
        #     and (np.abs(s[2:]) < 100).all()
        #     and (height > 0.7)
        #     and (abs(ang) < 0.2)
        # )
        # ob = self._get_obs()

        # if self.render_mode == "human":
        #     self.render()
        # return ob, reward, terminated, False, {}
        distance_before = np.abs(self.task - self.get_body_com('foot')[0])
        state, reward, terminated, _, info = super().step(a)
        state = np.append(self.get_body_com("foot")[0], state)
        distance_after = np.abs(self.task - self.get_body_com('foot')[0])
        ctrl_cost = 1e-3 * np.square(a).sum()
        healthy_penalty = 0
        bonus = 0
        info['reached_goal'] = False
        info['x_pos'] = self.get_body_com('foot')[0]
        if distance_after < 0.05 and not terminated:
            info['reached_goal'] = True
            bonus = 10

        # Compute the new reward based on the distance to the goal
        # Assuming the horizontal position of the Hopper is given by the x-coordinate (usually the first state variable)
        # distance_to_goal = np.abs(self.get_body_com('foot')[0] - self.task)
        step_reward = -(distance_after - distance_before)/self.dt
        healthy_reward = 1
        # reward = - distance_after
        # if not terminated:
        #     healthy_reward = 1
        if terminated:
            healthy_penalty = -10

        # rewards = step_reward + healthy_reward * self.healthy_scale   # Reward is negative distance to the goal
        # rewards = -distance_after/self.task+ healthy_reward * self.healthy_scale   # Reward is negative distance to the goal
        rewards = step_reward + healthy_reward * self.healthy_scale # Reward is negative distance to the goal
        reward = rewards - ctrl_cost
        # reward = -distance_after

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
        

    
