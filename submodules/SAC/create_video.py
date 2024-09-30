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

from torch import from_numpy

from configs.transfer_functions_config import transfer_config
from model import PolicyNetwork as TransferFunction
from meta_envs.toy_goal import Toy1D
from meta_envs.mujoco.cheetah import HalfCheetahEnvExternalTask, HalfCheetahGoal
from smrl.policies.exploration import RandomPolicy, RandomMemoryPolicy, MultiRandomMemoryPolicy
from sac_envs.walker import WalkerGoal
from sac_envs.hopper import HopperGoal
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from sac_envs.hopper_multi import HopperMulti
from sac_envs.walker_multi import WalkerMulti
from sac_envs.ant_multi import AntMulti
from sac_envs.walker_multi import WalkerMulti
from typing import List, Any, Dict, Callable

from mrl_analysis.video.video_creator import VideoCreator
import imageio
# from experiments_configs.half_cheetah_multi_env import config as env_config
# from experiments_configs.walker_multi import config as env_config
import json

# from transfer_function.transfer_configs.hopper_config import config

config = dict(
    experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    experiment_name = 'hopper_dt0.01_skipframe1',
    epoch = 7700,
)
with open(config['experiments_repo'] + config['experiment_name'] + '/config.json', 'r') as file:
        env_config = json.load(file)
if env_config['env'] == 'hopper':
        env = HopperGoal()
elif env_config['env'] == 'walker':
    env = WalkerGoal()
elif env_config['env'] == 'half_cheetah_multi':
    env = HalfCheetahMixtureEnv(env_config)
# elif config['env'] == 'half_cheetah_multi_vel':
#     env = HalfCheetahMixtureEnvVel()
elif env_config['env'] == 'hopper_multi':
    env = HopperMulti(env_config)
elif env_config['env'] == 'walker_multi':
    env = WalkerMulti(env_config)
elif env_config['env'] == 'ant_multi':
    env = AntMulti()

def _frames_to_video(video: cv2.VideoWriter, frames: List[np.ndarray], transform: Callable = None):
    """ Write collected frames to video file """
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        video.write(frame)

def _frames_to_gif(frames: List[np.ndarray], gif_path, transform: Callable = None):
    """ Write collected frames to video file """
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    with imageio.get_writer(gif_path, mode='I', fps=40) as writer:
        for i, frame in enumerate(frames):
            frame = frame.astype(np.uint8)  # Ensure the frame is of type uint8
            frame = np.ascontiguousarray(frame)
            # Apply transformation if any
            if transform is not None:
                frame = transform(frame)
            else:
                # Convert color space if no transformation provided
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            writer.append_data(frame)

def create_video():

    name_of_video = 'test'
    fps = 10
    video_creator = VideoCreator()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")

    env.render_mode = 'rgb_array'

    pretrained = config['experiments_repo']+config['experiment_name']+f"/models/policy_model/epoch_{config['epoch']}.pth"
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]
    transfer_function = TransferFunction(
        n_states=n_states,
        n_actions=n_actions,
        action_bounds=action_bounds,
        pretrained=pretrained
        )
    
    transfer_function = transfer_function.to(device)




    traj_len = 500
    obs = env.reset()[0]
    w=env.screen_width
    h=env.screen_height
    video_dir = "/home/ubuntu/juan/Meta-RL/evaluation/videos_of_transfer/"
    os.makedirs(os.path.dirname(video_dir), exist_ok=True)
    frames = []
    # simple_action = np.random.rand()
    # task = np.array([.0,0,0,-2.0,0])
    task = np.array([5.0,.0])
    env.update_task(task)
    # task = np.expand_dims(task, axis=0)
    task = from_numpy(task).float().to(device)
    for i in range(traj_len):
        # obs = np.expand_dims(obs, axis=0)
        obs = from_numpy(obs).float().to(device)
        action,_ = transfer_function.sample_or_likelihood(obs, task)
        # action = action[0]
        # action = env.action_space.sample()
        next_obs, reward, terminal, _, info = env.step(action.detach().cpu().numpy())
        # next_obs, reward, terminal, _, info = env.step(action)
        # if i == 10:
        #     simple_action = 0.3
        # if i == 30:
        #     simple_action = -0.3
        # if i == 200:
        #     task = torch.Tensor([6,0,0,0,0])
        if i == 200:
            task = np.array([.3, 0])
            env.update_task(task)
        #     # task = np.expand_dims(task, axis=0)
            task = from_numpy(task).float().to(device)
            print('change oof task')
        # if i == 600:
        #     task = np.array([.1, 0])
        #     env.update_task(task)
        #     # task = np.expand_dims(task, axis=0)
        #     task = from_numpy(task).float().to(device)
        #     print('change oof task')
        print('X:',i, next_obs[-3:], reward)
        print('terminal:', terminal)
        print('vel:', env.sim.data.qvel[0])
        # if np.abs(env.sim.data.qpos[2])>2*np.pi:
        #     break
        # if i == 250:
        #     task = np.array([1.0])
        #     task = from_numpy(task).float().to(device)

        # if i == 500:
        #     task = np.array([-1.0])
        #     task = from_numpy(task).float().to(device)
        # if terminal:
        #     break

        # kwargs = {}
        # if w is not None: kwargs['width'] = w
        # if h is not None: kwargs['height'] = h
        image = env.render()
        frames.append(image)


        obs = next_obs


        
        # if environment_steps % env_reset_interval == 0 or terminal:
        #     env.sample_task()
        #     obs, _ = env.reset()


    # Initialize VideoWriter (only once)
    size = frames[0].shape

    # Save to corresponding repo
    save_as = config['experiments_repo']+config['experiment_name'] + f'/epoch_{config["epoch"]}.mp4'
    # video = cv2.VideoWriter(save_as, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    # Write frames to video
    # _frames_to_video(video, frames)
    # video.release()
    _frames_to_gif(frames, save_as)

    # Save to place for easier scp
    save_as = config['experiments_repo'] + 'video.mp4'
    # video = cv2.VideoWriter(save_as, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    # Write frames to video
    # _frames_to_video(video, frames)
    # video.release()
    _frames_to_gif(frames, save_as)
    print("DONE")



if __name__ == "__main__":
    create_video()