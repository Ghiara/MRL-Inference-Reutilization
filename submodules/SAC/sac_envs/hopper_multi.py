import gym
from sac_envs.base_envs.hopper import HopperEnv
from gym.utils.ezpickle import EzPickle
from typing import List, Tuple
import numpy as np
import mujoco_py
from meta_envs.pygame_rendering import PygameRenderer
from gym.spaces import Box


class HopperMulti(HopperEnv):
    # rewrite this to give as input the dictionary
    def __init__(self, config, healthy_scale = 1, render_mode: str = 'rgb_array', *args, **kwargs):
        super().__init__(render_mode=render_mode, *args, **kwargs)
        self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64
            )
        self.healthy_scale = healthy_scale
        self.screen_height = 400
        self.screen_width = 400
        self.config = config
        self.task = self.sample_task()

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

    def step(self, action, healthy_scale=None):
        if healthy_scale is not None:
            self.healthy_scale = healthy_scale
        # Task is 5 dimensional -> it can either be jump, go forward/backward, rotation, velocity and flip velocity
        # this can be just q pos and qvel[0]
        # change task after some steps


        xposbefore = self.get_body_com('torso')[0]
        distance_before = np.abs(self.task[0] - self.get_body_com('torso')[0])
        state, reward, terminated, _, info = super().step(action)
        state = np.append(self.get_body_com("torso")[0], state)
        distance_after = np.abs(self.task[0] - self.get_body_com('torso')[0])
        xposafter = self.get_body_com('torso')[0]

        ob = self._get_obs()


        posafter, height, ang = self.sim.data.qpos[0:3]
        s = self.state_vector()
        terminated = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < 0.2)
        )
        healthy_penalty = 0
        if terminated:
            healthy_penalty = -10

        healthy_reward = 0.7

        # if task[3]!=0:  # 'velocity'
        #     reward_run = - np.abs(xvelafter[0] - self.task_specification)
        #     reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
        #     reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)
        
        if self.base_task in [self.config.get('tasks',{}).get('goal_front'), self.config.get('tasks',{}).get('goal_back')]:  # 'goal
            # reward_run = -(distance_after - distance_before)/np.abs(self.task[0])
            reward_run = -np.abs(xposafter-self.task[self.base_task])/np.abs(self.norm)
            # reward_run = reward_run.clip(-2,0)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward_ctrl = 0
            healthy_reward = 1
            if terminated:
                healthy_penalty = -10
            rewards = reward_run + healthy_reward * self.healthy_scale # Reward is negative distance to the goal
            reward = rewards + reward_ctrl

        elif self.base_task in [self.config.get('tasks',{}).get('forward_vel'), self.config.get('tasks',{}).get('backward_vel')]: # velocity
            healthy_reward = 1
            forward_vel = (xposafter - xposbefore) / self.dt
            reward_run = -1.0 * np.abs(forward_vel - self.task[self.base_task]) / np.abs(self.norm)
            # reward_run = reward_run.clip(-2,0)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward_ctrl = 0
            reward = reward_ctrl * 1.0 + reward_run + healthy_reward * self.healthy_scale
            # if np.abs(forward_vel - self.task[3]) < 0.1:
            #     self.reached_goal += 1
            #     if self.reached_goal == 20:
            #         reward+=1
        else:
            raise RuntimeError("base task not recognized")
        
        return ob, reward, terminated, False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
                                      true_task=self.task)

    # from pearl rlkit
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()


    def update_task(self, task):
        self.task = task

    def update_base_task(self, base_task):
        self.base_task = base_task

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
    
    
    def sample_task(self, test=False, task=None):
        self.task = np.zeros(max(self.config['tasks'].values())+1)
        # {'velocity_forward': 0, 'velocity_backward': 1, 'goal_forward': 4, 'goal_backward': 5, 
        # 'flip_forward': 6, 'stand_front': 3, 'stand_back': 2, 'jump': 7, flip_backward = 8,
        # 'direction_forward': -1, 'direction_backward': -1, 'velocity': -1}
        base_task = np.random.choice(list(self.config['tasks'].keys()))
        self.base_task = self.config.get('tasks',{}).get(base_task)
        mult = np.random.random()
        if task:
            base_task = task['base_task']
            self.base_task = self.config.get('tasks',{}).get(base_task)
            mult = task['specification']
        if base_task == 'goal_front':
            if test:
                self.task[self.base_task] = mult * (self.config['max_goal'][1] - self.config['max_goal'][0]) + self.config['max_goal'][0]
                self.norm = self.task[self.base_task]
            else:
                self.norm = mult + 0.5
                self.task[self.base_task] = self.sim.data.qpos[0] + self.norm
        elif base_task == 'goal_back':
            if test:
                self.task[self.base_task] = - (mult * (self.config['max_goal'][1] - self.config['max_goal'][0]) + self.config['max_goal'][0])
                self.norm = self.task[self.base_task]
            else:
                self.norm = mult + 0.5
                self.task[self.base_task] = self.sim.data.qpos[0] - self.norm
        elif base_task == 'forward_vel':
            self.task[self.base_task] = mult * (self.config['max_vel'][1] - self.config['max_vel'][0]) + self.config['max_vel'][0]
            self.norm = self.task[self.base_task]
        elif base_task == 'backward_vel':
            self.task[self.base_task] = - (mult * (self.config['max_vel'][1] - self.config['max_vel'][0]) + self.config['max_vel'][0])
            self.norm = self.task[self.base_task]
        elif base_task == 'stand_front':
            self.task[self.base_task] = mult * (self.config['max_rot'][1] - self.config['max_rot'][0]) + self.config['max_rot'][0]
            self.norm = self.task[self.base_task]
        elif base_task == 'stand_back':
            self.task[self.base_task] = - (mult * (self.config['max_rot'][1] - self.config['max_rot'][0]) + self.config['max_rot'][0])
            self.norm = self.task[self.base_task]
        # elif base_task == 6: # instead of rotation velocity, sample how many flips
        #     sign = np.random.choice(np.array([1,2]))
        #     self.task[0] = (-1)**sign*(mult * (self.config['max_rot_vel'][1] - self.config['max_rot_vel'][0]) + self.config['max_rot_vel'][0])
        #     # flips = np.random.choice(np.array([1,2,3]))
        #     # self.task[4] = -2*np.pi*flips
        # elif base_task == 8:
        #     self.task[0] = -(mult * (self.config['max_rot_vel'][1] - self.config['max_rot_vel'][0]) + self.config['max_rot_vel'][0])
        elif base_task == 'jump':
            self.task[self.base_task] = mult * (self.config['max_jump'][1] - self.config['max_jump'][0]) + self.config['max_jump'][0]
            self.norm = self.task[self.base_task]
        else: 
            print('Task not found')
        return self.task