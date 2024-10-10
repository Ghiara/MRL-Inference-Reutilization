
from agent import SAC
import json
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from sac_envs.hopper_multi import HopperMulti
from sac_envs.walker_multi import WalkerMulti
from sac_envs.ant_multi import AntMulti
import torch
import cv2
from typing import List, Any, Dict, Callable
import imageio
from typing import Tuple

from tqdm import tqdm
import os

'''
Choose the experiment to run here
'''
from experiments_configs.half_cheetah_multi_env import config


import matplotlib.pyplot as plt
import pandas as pd

import os
from datetime import datetime
import pytz

from mrl_analysis.utility.data_smoothing import smooth_plot, smooth_fill_between
from vis_utils.logging import log_all, _frames_to_gif

saved_dict = False
record_video_every = 50


def train(env, agent, epochs, experiment_name, save_after_episodes, policy_update_steps, batch_size=config['batch_size'], path=os.path.join(os.getcwd(), 'output/low_level_policy')):

    # For logging
    value_loss_history, q_loss_history, policy_loss_history, rew_history = [], [], [], []
    max_traj_len_start = config['max_traj_len']
    
    path=f'{path}/{experiment_name}'
    change_task = False
    traj_len = []

    os.makedirs(path, exist_ok=True)
    file_path = path + '/config.json'
    with open(file_path, "w") as json_file:
        json.dump(config, json_file)
    
    '''
    Used for curriculum learning
    '''
    restart_random = False
    # change_after = config['change_tasks_after']
    plot_every = config['plot_every']
    change_task = 0
    max_vel = False
    random_init = False


    for episode in range(1, epochs + 1):
        print("\033[1m" + f"Episode: {episode}" + "\033[0m")

        '''
        Initialize list for logging
        '''
        max_traj_len = max_traj_len_start
        rew = []
        value_loss, q_loss, policy_loss = [], [], []
        number_of_changes = 0
        env.reached_goal = 0
        batch_reward = []
        task_changes = 0

        '''
        Parameters used for curriculum learning
        '''
        if 'curriculum' in config:
            for i, change_epoch in enumerate(config['curriculum']['change_tasks_after']):

                if episode/change_epoch>1:
                    task_changes = config['curriculum']['changes_per_trajectory'][i]

            if episode%config['curriculum']['max_vel']==0:
                max_vel=True

            for j, max_steps_epoch in enumerate(config['curriculum']['max_steps_epochs']):
                if episode/max_steps_epoch>1:
                    max_traj_len = config['curriculum']['max_steps'][j]

            if episode/config['curriculum']['random_initialization']>=1:
                random_init = True

        for batch in tqdm(range(batch_size)):

            plot = False
            record_video = False
            if episode % record_video_every == 0 and batch==batch_size-1:
                frames = []
                image_info = dict(reward = [],
                obs = [],
                base_task = [],
                task = [],
                action = [])
                plot = True
                record_video = True

            # if episode > random_restart_after: 
            #     restart_random = True
            # if restart_random:
            #     state = env.random_reset()[0]
            # else:
            '''
            Used to randomly initialize the agent's position
            '''
            if random_init:
                state = env.random_reset(x_pos_range=[-50,50])[0]
            else:
                state = env.reset()[0]
            done = 0
            task = env.sample_task()

            '''
            Part of the curriculum learning
            '''
            if not max_vel and env.base_task in [env.config.get('tasks',{}).get('forward_vel'), env.config.get('tasks',{}).get('backward_vel')]:
                task = task/2
                                                 
            env.update_task(task)

            episode_reward = 0
            j = 0

            '''
            Run trajectory of len max_traj_len
            '''
            while not done and j<max_traj_len:
                if batch == 0:
                    save = True
                else: save = False

                '''
                choose action and perform it
                '''
                action = agent.choose_action(state, task)
                next_state, reward, done, _, info = env.step(action)
                # reward = reward.clip(-5, 2)

                '''
                Train after every step with random sample from buffer
                '''
                agent.store(state, reward, done, action, next_state, task)
                losses = agent.train(episode, save)
                value_loss.append(losses[0])
                policy_loss.append(losses[2])
                q_loss.append(losses[1])
                
                '''
                Save the position for plotting trajectories
                '''
                if env.base_task == env.config.get('tasks',{}).get('goal_front'):
                    curr_state = env.sim.data.qpos[0]
                elif env.base_task == env.config.get('tasks',{}).get('goal_back'):
                    curr_state = env.sim.data.qpos[0]
                elif env.base_task == env.config.get('tasks',{}).get('forward_vel'):
                    curr_state = env.sim.data.qvel[0]
                elif env.base_task == env.config.get('tasks',{}).get('backward_vel'):
                    curr_state = env.sim.data.qvel[0]
                elif env.base_task == env.config.get('tasks',{}).get('stand_front'):
                    curr_state = env.sim.data.qpos[2]
                elif env.base_task == env.config.get('tasks',{}).get('stand_back'):
                    curr_state = env.sim.data.qpos[2]
                elif env.base_task == env.config.get('tasks',{}).get('jump'):
                    curr_state = env.sim.data.qvel[1]

                '''
                If the agent has reached the goal change task with prob 0.2. Part of Curriculum Learning
                '''
                if np.abs(curr_state - env.task[env.base_task])<0.1 and np.random.random()>0.8:
                    change_task = True
                    number_of_changes+=1
                # if np.random.random()<0.05:
                #     change_task = True
                if task_changes:
                    if j%(max_traj_len//task_changes) == 0 and j!=0:
                        task = env.sample_task()
                if change_task:
                    # if j%(max_traj_len//change_task) == 0 and j!=0:
                    task = env.sample_task()
                    change_task = False
                
                episode_reward += reward
                state = next_state
                rew.append(reward)

                j+=1
                if record_video:
                    image = env.render()
                    frames.append(image)
                    image_info['reward'].append(reward)
                    image_info['obs'].append([env.sim.data.qpos[0], env.sim.data.qvel])
                    image_info['base_task'].append(env.base_task)
                    image_info['task'].append(env.task)
                    image_info['action'].append(action)
        
            
            batch_reward.append(episode_reward)
            if record_video:
                save_video_path = f'{path}/videos/{episode}_{batch}.mp4'
                _frames_to_gif(frames, image_info, save_video_path)

        traj_len.append(j)
        rew_history.append(np.mean(batch_reward))
        value_loss_history.append(np.mean(value_loss))
        policy_loss_history.append(np.mean(policy_loss))
        q_loss_history.append(np.mean(q_loss))

        print(experiment_name)
        print('Trajectory len:', j)
        print('End position:', env.sim.data.qpos[0], 'End vel:', env.sim.data.qvel[0])
        print('Task:', task, 'Base task:', env.base_task, '\tMean Task changes:', number_of_changes/batch_size)
        print('Mean reward:', np.mean(batch_reward))
        # print('Total reward:', episode_reward)
        # log(episode, start_time, episode_reward, value_loss, q_loss, policy_loss, len(agent.memory))

        if episode % save_after_episodes == 0 and episode!=0:
            log_all(agent, path, q_loss_history, policy_loss_history, rew_history, traj_len, episode)
            data = {
                'Q Loss History': q_loss_history,
                'Policy Loss History': policy_loss_history,
                'Reward History': rew_history,
                'Trajectory Length': traj_len
            }
            df = pd.DataFrame(data)

            # Save to CSV
            csv_path = f'{path}/progress.csv'
            df.to_csv(csv_path, index=False)

if __name__ == "__main__":

    '''
    Load environment with agent to be trained
    '''
    if config['env'] == 'half_cheetah_multi':
        env = HalfCheetahMixtureEnv(config)
    elif config['env'] == 'hopper_multi':
        env = HopperMulti(config)
    elif config['env'] == 'walker_multi':
        env = WalkerMulti(config)
    elif config['env'] == 'ant_multi':
        env = AntMulti()
    else:
        print('Invalid argument for environment name... Instantiating hopper env instead')
        env = HopperGoal()

    env.render_mode = 'rgb_array'

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]
    epochs = config['epochs']
    batch_size = config['batch_size']
    save_after_episodes = config['save_after_episodes']
    print(f"Number of states:{n_states}\n"
          f"Number of actions:{n_actions}\n"
          f"Action boundaries:{action_bounds}")
    
    experiment_name = config['experiment_name']
    reward_scale = config['reward_scale']           # If environment is humanoid: reward_scale = 20
    
    if 'pretrained' not in config:
        config['pretrained'] = None
    agent = SAC(n_states=n_states,
                n_actions=n_actions,
                task_dim = max(config['tasks'].values())+1,
                hidden_layers_actor = config['hidden_layers_actor'],
                hidden_layers_critic = config['hidden_layers_critic'],
                memory_size=config['memory_size'],
                batch_size=config['batch_size_memory'],
                gamma=config['gamma'],
                alpha=config['alpha'],
                lr=config['lr'],
                action_bounds=action_bounds,
                reward_scale=reward_scale,
                pretrained=config['pretrained'])
    train(env, agent, epochs, experiment_name, save_after_episodes, batch_size=batch_size, policy_update_steps=config['policy_update_steps'])