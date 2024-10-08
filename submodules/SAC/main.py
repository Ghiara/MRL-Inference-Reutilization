
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

# Choose which experiment to run
from experiments_configs.half_cheetah_multi_env import config


import matplotlib.pyplot as plt
import pandas as pd

import os
from datetime import datetime
import pytz

from mrl_analysis.utility.data_smoothing import smooth_plot, smooth_fill_between

# TRAIN = False
saved_dict = False
record_video_every = 50

def log_all(agent, path, q1_loss, policy_loss, rew, traj_len, episode):
    '''
    # Save under structure:
    # - {os.getcwd()}/experiments_transfer_function/<name_of_experiment>
    #     - plots
    #         - mean_reward_history
    #         - qf_loss
    #         - policy_loss
    #     - models
    #         - transfer_function / policy_net
    #         - qf1
    #         - value
    '''

        # TODO: save both vf losses (maybe with arg)
    def save_plot(loss_history, name:str, path=f'{os.getcwd()}/evaluation/transfer_function/one-sided/', figure_size: Tuple[int,int] = (20, 10)):
        def remove_outliers_iqr(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = np.array(data)
            return data[(data >= lower_bound) & (data <= upper_bound)]
        
        def moving_average(data, window_size=10):
            # window = np.ones(int(window_size))/float(window_size)
            # return np.convolve(data, window, 'same')
            index_array = np.arange(1, len(data) + 1)
            data = pd.Series(data, index = index_array)
            return data.rolling(window=window_size).mean()
        def format_label(label):
            words = label.split('_')
            return ' '.join(word.capitalize() for word in words)
        
        # TODO: save both vf losses (maybe with arg)
        os.makedirs(path, exist_ok=True)
        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, np.arange(len(loss_history)), loss_history, label=format_label(name))
        axs.legend()
        axs.set_xlabel('Train epochs')

        # plt.figure()
        # # Plotting the loss
        # # plt.plot(smoothed_loss, color='blue')
        # # plt.plot(loss_history, color='blue', alpha=0.3)
        # plt.title('Loss over Time')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        fig.savefig(os.path.join(path,name+'.png'))

        plt.close()

    # Save networks
    curr_path = path + '/models/policy_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'epoch_{episode}.pth'
    if episode % 100 == 0:
        torch.save(agent.policy_network.cpu(), save_path)
    curr_path = path + '/models/vf1_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'epoch_{episode}.pth'
    if episode % 100 == 0:
        torch.save(agent.q_value_network1.cpu(), save_path)
    curr_path = path + '/models/vf2_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'epoch_{episode}.pth'
    if episode % 100 == 0:
        torch.save(agent.q_value_network2.cpu(), save_path)
    curr_path = path + '/models/value_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'epoch_{episode}.pth'
    if episode % 100 == 0:
        torch.save(agent.value_network.cpu(), save_path)
    agent.q_value_network1.cuda() 
    agent.q_value_network2.cuda()
    agent.value_network.cuda()
    agent.policy_network.cuda() 

    # Save plots
    path_plots = path + '/plots/'
    save_plot(q1_loss, name='vf_loss', path=path_plots)
    save_plot(rew, name='reward_history', path=path_plots)
    save_plot(policy_loss, name='policy_loss', path=path_plots)
    save_plot(traj_len, name='traj_len', path=path_plots)

# def sample_task():
#     task = np.zeros(5)
#     # {'velocity_forward': 0, 'velocity_backward': 1, 'goal_forward': 4, 'goal_backward': 5, 
#     # 'flip_forward': 6, 'stand_front': 3, 'stand_back': 2, 'jump': 7, flip_backward = 8,
#     # 'direction_forward': -1, 'direction_backward': -1, 'velocity': -1}
#     base_task = np.random.choice(np.arange(0,8))
#     mult = np.random.random()
#     if base_task == 4:
#         task[0] = mult * (config['max_goal'][1] - config['max_goal'][0]) + config['max_goal'][0]
#     elif base_task == 5:
#         task[0] = - (mult * (config['max_goal'][1] - config['max_goal'][0]) + config['max_goal'][0])
#     elif base_task == 0:
#         task[3] = mult * (config['max_vel'][1] - config['max_vel'][0]) + config['max_vel'][0]
#     elif base_task == 1:
#         task[3] = - (mult * (config['max_vel'][1] - config['max_vel'][0]) + config['max_vel'][0])
#     elif base_task == 2:
#         task[2] = mult * (config['max_rot'][1] - config['max_rot'][0]) + config['max_rot'][0]
#     elif base_task == 3:
#         task[2] = - (mult * (config['max_rot'][1] - config['max_rot'][0]) + config['max_rot'][0])
#     elif base_task == 6: # instead of rotation velocity, sample how many flips
#         # task[4] = mult * (config['max_rot_vel'][1] - config['max_rot_vel'][0]) + config['max_rot_vel'][0]
#         flips = np.random.choice(np.array([1,2,3]))
#         task[4] = -2*np.pi*flips
#     elif base_task == 8:
#         task[4] = -(mult * (config['max_rot_vel'][1] - config['max_rot_vel'][0]) + config['max_rot_vel'][0])
#     elif base_task == 7:
#         task[1] = mult * (config['max_jump'][1] - config['max_jump'][0]) + config['max_jump'][0]
#     return task

def _frames_to_gif(frames: List[np.ndarray], info, gif_path, transform: Callable = None):
    """ Write collected frames to video file """
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    with imageio.get_writer(gif_path, mode='I', fps=10) as writer:
        for i, frame in enumerate(frames):
            frame = frame.astype(np.uint8)  # Ensure the frame is of type uint8
            frame = np.ascontiguousarray(frame)
            cv2.putText(frame, 'reward: ' + str(info['reward'][i]), (0, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            cv2.putText(frame, 'obs: ' + str(info['obs'][i]), (0, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            cv2.putText(frame, 'action: ' + str(info['action'][i]), (0, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            cv2.putText(frame, 'task: ' + str(info['base_task'][i]), (0, 75), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            cv2.putText(frame, 'task: ' + str(info['task'][i]), (0, 95), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            # Apply transformation if any
            if transform is not None:
                frame = transform(frame)
            else:
                # Convert color space if no transformation provided
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            writer.append_data(frame)

def train(env, agent, epochs, experiment_name, save_after_episodes, policy_update_steps, batch_size=config['batch_size']):

    # For logging
    value_loss_history, q_loss_history, policy_loss_history, rew_history = [], [], [], []
    max_traj_len_start = config['max_traj_len']
    # random_restart_after = config['random_restart_after']

    change_task = False
    traj_len = []


    path = f'{os.getcwd()}/experiments_transfer_function/{experiment_name}/'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_path = path + '/config.json'
    with open(file_path, "w") as json_file:
        json.dump(config, json_file)
        
    restart_random = False
    # change_after = config['change_tasks_after']
    plot_every = config['plot_every']
    change_task = 0
    max_vel = False
    random_init = False
    for episode in range(1, epochs + 1):
        print("\033[1m" + f"Episode: {episode}" + "\033[0m")

        # max_traj_len = max_traj_len_start + episode//3
        max_traj_len = max_traj_len_start

        rew = []
        value_loss, q_loss, policy_loss = [], [], []
        number_of_changes = 0
        env.reached_goal = 0
        batch_reward = []
        task_changes = 0
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

            # if episode > random_restart_after: 
            #     restart_random = True
            # if restart_random:
            #     state = env.random_reset()[0]
            # else:
            if random_init:
                state = env.random_reset(x_pos_range=[-50,50])[0]
            else:
                state = env.reset()[0]

            done = 0

            #                 #
            # Simple task env #
            #                 #
            # task = np.array([np.random.rand() * 2 - 1])

            #                        #
            # Multi task environment #
            #                        #
            # task = env.sample_task(test=True)
            task = env.sample_task()
            # task = np.array([-10.0])

            if not max_vel and env.base_task in [env.config.get('tasks',{}).get('forward_vel'), env.config.get('tasks',{}).get('backward_vel')]:
                task = task/2
                                                 
            env.update_task(task)

            episode_reward = 0
            j = 0

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
            while not done and j<max_traj_len:
                if batch == 0:
                    save = True
                else: save = False

                # change task randomly in the middle of the trajectory
                # if episode>1 and np.random.random()>0.6 and i == 250:
                # if episode>200 and i % 50 == 0 and i != 0 and np.random.random()>0.2:
                #     task = env.sample_task()
                #     # task = np.array([-10.0])
                #     task_changes += 1
                #     env.update_task(task)

                action = agent.choose_action(state, task)
                next_state, reward, done, _, info = env.step(action)
                # reward = reward.clip(-5, 2)
                agent.store(state, reward, done, action, next_state, task)
                losses = agent.train(episode, save)
                value_loss.append(losses[0])
                policy_loss.append(losses[2])
                q_loss.append(losses[1])
                
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
                # elif base_task == 6: # instead of rotation velocity, sample how many flips
                #     sign = np.random.choice(np.array([1,2]))
                #     self.task[0] = (-1)**sign*(mult * (self.config['max_rot_vel'][1] - self.config['max_rot_vel'][0]) + self.config['max_rot_vel'][0])
                #     # flips = np.random.choice(np.array([1,2,3]))
                #     # self.task[4] = -2*np.pi*flips
                # elif base_task == 8:
                #     self.task[0] = -(mult * (self.config['max_rot_vel'][1] - self.config['max_rot_vel'][0]) + self.config['max_rot_vel'][0])
                elif env.base_task == env.config.get('tasks',{}).get('jump'):
                    curr_state = env.sim.data.qvel[1]
                # if 'reached_goal' in info:
                #     if info['reached_goal'] and change_task:
                #         task += np.array([np.random.rand()*2 - 1])
                #         task_changes += 1
                #         env.update_task(task)
                # if hasattr(env, 'reached_goal'):
                #     if env.reached_goal >= 20:
                #         task = env.sample_task()
                #         env.update_task(task)
                #         task_changes+=1
                #         env.reached_goal = 0
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


                # if env.base_task in [4,5] and j % change_after == 0 and j != 0:
                #     task = env.sample_task()
                #     env.update_task(task)
                # elif j % change_after*10 == 0 and j != 0:
                #     task = env.sample_task()
                #     env.update_task(task)


                episode_reward += reward
                state = next_state
                rew.append(reward)
                # if np.abs(episode_reward)>2000:
                #     pass
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
                # fps=10
                # size = frames[0].shape
                # video = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
                # Write frames to video
                _frames_to_gif(frames, image_info, save_video_path)
                # video.release()


        # if episode%500 == 0 and change_after>50:
        #     change_after-=50
        # for _ in tqdm(range(policy_update_steps)):
        #     losses = agent.train(episode, save)
        #     value_loss.append(losses[0])
        #     policy_loss.append(losses[2])
        #     q_loss.append(losses[1])

        

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

    # if config['env'] == 'hopper':
    #     env = HopperGoal()
    # elif config['env'] == 'walker':
    #     env = WalkerGoal()
    if config['env'] == 'half_cheetah_multi':
        env = HalfCheetahMixtureEnv(config)
    # elif config['env'] == 'half_cheetah_pos_add':
    #     env = HalfCheetahPosAdd(config)
    # elif config['env'] == 'half_cheetah_multi_vel':
    #     env = HalfCheetahMixtureEnvVel()
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