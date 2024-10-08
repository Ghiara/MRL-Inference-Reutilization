import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import cv2

from configs.transfer_functions_config import transfer_config
from smrl.trainers.transfer_function import TransferFunction, RewardPredictor
from meta_envs.toy_goal import Toy1D
from meta_envs.mujoco.cheetah import HalfCheetahEnvExternalTask, HalfCheetahGoal
from smrl.policies.exploration import RandomPolicy, RandomMemoryPolicy, MultiRandomMemoryPolicy
from typing import List, Any, Dict, Callable

from mrl_analysis.video.video_creator import VideoCreator

from transfer_function.transfer_configs.hopper_config import config

def _frames_to_video(video: cv2.VideoWriter, frames: List[np.ndarray], transform: Callable = None):
    """ Write collected frames to video file """
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        video.write(frame)

name_of_video = 'test'
fps = 10
video_creator = VideoCreator()
transfer_function = TransferFunction(obs_dim=config['obs_dim_complex'], 
                                    act_simple_dim=config['act_simple_dim'], 
                                    hidden_sizes=config['hidden_sizes_transfer'], 
                                    act_complex_dim=config['act_complex_dim'],
                                    pretrained=config['pretrained_transfer_function']
                                    )

agent = MultiRandomMemoryPolicy(action_dim=1, action_update_interval=25, std_mean_range=(0.05,0.1)) # TODO: try differnet values
env_args = []
env_kwargs = {
    "n_train_tasks": 100,
    "n_eval_tasks": 25,
    "change_steps": 500,
    "task_generation_mode": 'random',
}
simple_env = Toy1D(*env_args, **env_kwargs)

env_kwargs = {
    "n_train_tasks": 100,
    "n_eval_tasks": 25,
    "change_prob": 0,
    "render_mode": 'rgb_array',
    "one_sided_tasks": True,
}
#TODO: load with variant.json
env = config['environment']


traj_len = 200
obs = env.reset()[0]
simple_obs = simple_env.reset()[0]
w=env.screen_width
h=env.screen_height
video_dir = "{os.getcwd()}/evaluation/videos_of_transfer/"
save_as = video_dir + name_of_video + '.mp4'
os.makedirs(os.path.dirname(video_dir), exist_ok=True)
frames = []
# simple_action = agent.get_action(simple_env.observation)[0][0]
simple_action = np.array([1.0])
env.update_task(simple_action)
for i in range(traj_len):
    action = transfer_function.get_action(torch.tensor(obs), torch.tensor(simple_action), return_dist=False)
    # action = env.action_space.sample()
    next_obs, reward, terminal, _, ifo = env.step(action.detach().cpu().numpy())
    # next_obs, reward, terminal, _, info = env.step(action)
    # if i == 10:
    #     simple_action = 0.3
    # if i == 30:
    #     simple_action = -0.3
    # if i == 60:
    #     simple_action = -1.0
    if i == 200:
        simple_action = np.array([1.0])
        env.update_task(simple_action)
        print('change oof task')
    print('X:',i, next_obs[0], reward)
    print('terminal:', terminal)
    # if i == 125:
    #     simple_action = np.array([1.0])
    # if terminal:
    #     break

    kwargs = {}
    if w is not None: kwargs['width'] = w
    if h is not None: kwargs['height'] = h
    image = env.render('rgb_array', **kwargs)
    frames.append(image)


    obs = next_obs


    
    # if environment_steps % env_reset_interval == 0 or terminal:
    #     env.sample_task()
    #     obs, _ = env.reset()


# Initialize VideoWriter (only once)
size = frames[0].shape
video = cv2.VideoWriter(save_as, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)

# Write frames to video
_frames_to_video(video, frames)
video.release()
print("DONE")