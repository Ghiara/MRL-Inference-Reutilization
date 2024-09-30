import numpy as np
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.spaces.box import Box
from gym import utils
import mujoco_py
import os


class HalfCheetahPosAdd(HalfCheetahEnv, utils.EzPickle):
    def __init__(self, config, healthy_scale = 1, render_mode: str = 'rgb_array', *args, **kwargs):
        self.healthy_scale = healthy_scale
        self.screen_height = 400
        self.screen_width = 400
        self.task = np.array([1])
        self.termination_possible = False
        super().__init__(frame_skip=5, *args, **kwargs)
        self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64
            )
        self.reached_goal = 0
        self.config = config

# class HalfCheetahMixtureEnv(HalfCheetahEnv, utils.EzPickle):
#     def __init__(self, healthy_scale = 1, render_mode: str = 'rgb_array', *args, **kwargs):
#         super().__init__(render_mode=render_mode,*args, **kwargs)
#         self.observation_space = Box(
#                 low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64
#             )
#         self.healthy_scale = healthy_scale
#         self.screen_height = 400
#         self.screen_width = 400
#         self.task = 0
#         self.termination_possible = False

    def step(self, action):
        # Task is 5 dimensional -> it can either be jump, go forward/backward, rotation, velocity and flip velocity
        # this can be just q pos and qvel[0]
        # change task after some steps

        xposbefore = np.copy(self.sim.data.qpos)
        try:
            result = super().step(action)
        except:
            raise RuntimeError("Simulation error, common error is action = nan")

        xposafter = np.copy(self.sim.data.qpos)
        xvelafter = np.copy(self.sim.data.qvel)

        ob = self._get_obs()
        # xposafter = ob[-3:]
        # xvelafter = ob[8:11]

        # if task[3]!=0:  # 'velocity'
        #     reward_run = - np.abs(xvelafter[0] - self.task_specification)
        #     reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
        #     reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)
        
        if self.base_task == 4 or self.base_task == 5:  # 'goal'
            reward_run = - np.abs(xposafter[0] - self.task[0])
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task[0])
            # if np.abs(xposafter[0] - self.task[0]) < 0.1:
            #     self.reached_goal += 1
            #     if self.reached_goal == 20:
            #         reward+=1
        elif self.base_task == 8:  # 'flipping'   distance tp -2*pi
            # reward_run = - np.abs(xvelafter[2] - self.task[4])
            # reward_run = -np.abs(xposafter[2] - self.task[4])
            # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            # reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task[4])
            # if np.abs(xposafter[2] - self.task[4]) < 5/360*2*np.pi:
            #     self.reached_goal += 1
            #     if self.reached_goal == 20:
            #         reward+=1
            reward_run = - np.abs(xvelafter[2] - self.task[0])
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task[4])

        elif self.base_task == 2 or self.base_task == 3:  # 'stand_up'
            reward_run = - np.abs(xposafter[2] - self.task[0])
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task[2])
            if np.abs(xposafter[2] - self.task[2]) < 5/360*2*np.pi:
                self.reached_goal += 1
                if self.reached_goal == 20:
                    reward+=1

        elif self.base_task == 7:  # 'jump'
            reward_run = - np.abs(np.abs(xvelafter[1]) - self.task[0])
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task[1])
            if np.abs(xvelafter[1]) - self.task[1] < 0.1:
                self.reached_goal += 1
                if self.reached_goal == 20:
                    reward+=1

        # elif self.base_task == 4 or self.base_task == 5:  # 'direction'
        #     reward_run = (xposafter[0] - xposbefore[0]) / self.dt * np.sign(self.task[0])
        #     reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
        #     reward = reward_ctrl * 1.0 + reward_run

        elif self.base_task == 0 or self.base_task == 1: # velocity
            forward_vel = xvelafter[0]
            reward_run = -1.0 * np.abs(forward_vel - self.task[0]) / np.abs(self.task[0])
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run
            # if np.abs(forward_vel - self.task[3]) < 0.1:
            #     self.reached_goal += 1
            #     if self.reached_goal == 20:
            #         reward+=1
        else:
            raise RuntimeError("base task not recognized")


        # if np.abs(reward_run)<0.2:
        #     done = True

        # print(str(self.base_task) + ": " + str(reward))
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
            done = not notdone
        else:
            done = False
        return ob, reward, done, False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
                                      true_task=self.task)

    # from pearl rlkit
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    # def reset_model(self):
    #     # reset changepoint
    #     self.positive_change_point = self.positive_change_point_basis + np.random.random() * self.change_point_interval
    #     self.negative_change_point = self.negative_change_point_basis - np.random.random() * self.change_point_interval

    #     # reset tasks
    #     self.base_task = self._task['base_task']
    #     self.task_specification = self._task['specification']
    #     self.recolor()

    #     # standard
    #     qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    # def get_image(self, width=256, height=256, camera_name=None):
    #     if self.viewer is None or type(self.viewer) != mujoco_py.MjRenderContextOffscreen:
    #         self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim)
    #         self.viewer_setup()
    #         self._viewers['rgb_array'] = self.viewer

    #     # use sim.render to avoid MJViewer which doesn't seem to work without display
    #     return self.sim.render(
    #         width=width,
    #         height=height,
    #         camera_name=camera_name,
    #     )

    # def viewer_setup(self):
    #     self.viewer.cam.type = 2
    #     self.viewer.cam.fixedcamid = 0

    # def change_task(self, spec):
    #     self.base_task = spec['base_task']
    #     self.task_specification = spec['specification']
    #     self._goal = spec['specification']
    #     self.color = spec['color']
    #     self.recolor()

    # def recolor(self):
    #     geom_rgba = self._init_geom_rgba.copy()
    #     rgb_value = self.color
    #     geom_rgba[1:, :3] = np.asarray(rgb_value)
    #     self.model.geom_rgba[:] = geom_rgba

    def update_task(self, task):
        self.task = task

    def reset(self):
        obs = super().reset()
        # new_obs = np.append(self.get_body_com("torso")[0], obs[0])
        new_obs = self._get_obs()
        return new_obs, {}
    
    def random_reset(self, x_pos_range=[-10,10], x_vel_range=[-0.1,0.1]):
        obs = super().reset()
        # new_obs = np.append(self.get_body_com("torso")[0], obs[0])

        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        qpos[0] = np.random.random() * (x_pos_range[1] - x_pos_range[0]) + x_pos_range[0]
        qvel[0] = np.random.random() * (x_vel_range[1] - x_vel_range[0]) + x_vel_range[0]
        self.set_state(qpos, qvel)

        new_obs = self._get_obs()
        return new_obs, {}
    
    def sample_task(self):
        self.task = np.zeros(self.config['task_dim'])
        # {'velocity_forward': 0, 'velocity_backward': 1, 'goal_forward': 4, 'goal_backward': 5, 
        # 'flip_forward': 6, 'stand_front': 3, 'stand_back': 2, 'jump': 7, flip_backward = 8,
        # 'direction_forward': -1, 'direction_backward': -1, 'velocity': -1}
        base_task = np.random.choice(list(self.config['tasks'].values()))
        self.base_task = base_task
        mult = np.random.random()
        if base_task == 4:
            self.task[0] = self.sim.data.qpos[0] + mult*2
        elif base_task == 5:
            self.task[0] = self.sim.data.qpos[0] - mult*2
        elif base_task == 0:
            self.task[0] = mult * (self.config['max_vel'][1] - self.config['max_vel'][0]) + self.config['max_vel'][0]
        elif base_task == 1:
            self.task[0] = - (mult * (self.config['max_vel'][1] - self.config['max_vel'][0]) + self.config['max_vel'][0])
        elif base_task == 2:
            self.task[0] = mult * (self.config['max_rot'][1] - self.config['max_rot'][0]) + self.config['max_rot'][0]
        elif base_task == 3:
            self.task[0] = - (mult * (self.config['max_rot'][1] - self.config['max_rot'][0]) + self.config['max_rot'][0])
        elif base_task == 6: # instead of rotation velocity, sample how many flips
            sign = np.random.choice(np.array([1,2]))
            self.task[0] = (-1)**sign*(mult * (self.config['max_rot_vel'][1] - self.config['max_rot_vel'][0]) + self.config['max_rot_vel'][0])
            # flips = np.random.choice(np.array([1,2,3]))
            # self.task[4] = -2*np.pi*flips
        elif base_task == 8:
            self.task[0] = -(mult * (self.config['max_rot_vel'][1] - self.config['max_rot_vel'][0]) + self.config['max_rot_vel'][0])
        elif base_task == 7:
            self.task[0] = mult * (self.config['max_jump'][1] - self.config['max_jump'][0]) + self.config['max_jump'][0]
        return self.task
