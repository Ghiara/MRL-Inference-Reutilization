import gym
from agent import SAC
import json
import time
import psutil
import mujoco_py
from torch.utils.tensorboard import SummaryWriter
# from play import Play
import numpy as np
from sac_envs.hopper import HopperGoal
from sac_envs.walker import WalkerGoal
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from sac_envs.half_cheetah_multi_vels import HalfCheetahMixtureEnvVel
from sac_envs.hopper_multi import HopperMulti
from sac_envs.walker_multi import WalkerMulti
from sac_envs.ant_multi import AntMulti
from sac_envs.half_cheetah_multi_toy import HalfCheetahMixtureToyEnv
from meta_rand_envs.toy1d_multi import Toy1dMultiTask
import torch
from create_video import create_video

# Choose which experiment to run
from experiments_configs.half_cheetah_multi_env_toy import config


import matplotlib.pyplot as plt

import os
from datetime import datetime
import pytz

# TRAIN = False
saved_dict = False

def log_all(agent, path, q1_loss, policy_loss, rew, traj_len, episode):
    '''
    # Save under structure:
    # - /home/ubuntu/juan/Meta-RL/experiments_transfer_function/<name_of_experiment>
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
    def save_plot(loss_history, name:str, path='/home/ubuntu/juan/Meta-RL/evaluation/transfer_function/one-sided/'):
        def moving_average(data, window_size):
            window = np.ones(int(window_size))/float(window_size)
            return np.convolve(data, window, 'same')
        
        window_size = 20
        smoothed_data = moving_average(loss_history, window_size)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        plt.figure()
        # Plotting the loss
        plt.plot(smoothed_data)
        plt.plot(loss_history, alpha=0.3)
        plt.title('Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(path+name+'.png')

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

def train(env, agent, epochs, experiment_name, save_after_episodes, policy_update_steps):

    # For logging
    value_loss_history, q_loss_history, policy_loss_history, rew_history = [], [], [], []
    max_traj_len = config['max_traj_len']

    change_task = False
    traj_len = []


    path = f'/home/ubuntu/juan/Meta-RL/experiments_transfer_function/{experiment_name}/'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_path = path + '/config.json'
    with open(file_path, "w") as json_file:
        json.dump(config, json_file)
        

    for episode in range(1, epochs + 1):
        print("\033[1m" + f"Episode: {episode}" + "\033[0m")
        state = env.reset()[0]
        episode_reward = 0
        done = 0

        #                 #
        # Simple task env #
        #                 #
        # task = np.array([np.random.rand() * 2 - 1])

        #                        #
        # Multi task environment #
        #                        #
        task = env.sample_task()
        # task = np.array([-10.0])
        env.update_task(task)
        i = 0
        rew = []
        value_loss, q_loss, policy_loss = [], [], []
        if episode > 225:
            change_task = True
        task_changes = 0
        env.reached_goal = 0
        while not done and i<max_traj_len:
            if i == 0:
                save = True
            else: save = False

            # change task randomly in the middle of the trajectory
            # if episode>1 and np.random.random()>0.6 and i == 250:
            if episode>3000 and i == 200:
                task = env.sample_task()
                # task = np.array([-10.0])
                task_changes = 1
                env.update_task(task)

            action = agent.choose_action(state, task)
            if i == 0:
                mean = np.random.normal(0, 1)
                var = np.random.exponential(0.3)
                    
            simple_action = np.random.normal(mean, var)
            simple_action = np.random.normal(loc=0, scale=1)
            simple_action = np.clip(simple_action, -1,1)

            next_state, reward, done, _, info = env.step(action, simple_action)
            agent.store(state, reward, done, action, next_state, task)
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
            episode_reward += reward
            state = next_state
            rew.append(reward)
            losses = agent.train(episode, save)
            value_loss.append(losses[0])
            policy_loss.append(losses[2])
            q_loss.append(losses[1])

            i+=1
        # for _ in range(policy_update_steps):

        traj_len.append(i)
        rew_history.append(np.mean(episode_reward))
        value_loss_history.append(np.mean(value_loss))
        policy_loss_history.append(np.mean(policy_loss))
        q_loss_history.append(np.mean(q_loss))

        print(experiment_name)
        print('Trajectory len:', i)
        print('End position:', info)
        print('Task:', task, '\tTask changes:', task_changes)
        print('Mean reward:', np.mean(rew))
        print('Total reward:', episode_reward)
        # log(episode, start_time, episode_reward, value_loss, q_loss, policy_loss, len(agent.memory))

        if episode % save_after_episodes == 0 and episode!=0:
            log_all(agent, path, q_loss_history, policy_loss_history, rew_history, traj_len, episode)

if __name__ == "__main__":

    if config['env'] == 'hopper':
        env = HopperGoal()
    elif config['env'] == 'walker':
        env = WalkerGoal()
    elif config['env'] == 'half_cheetah_multi':
        env = HalfCheetahMixtureEnv(config)
    elif config['env'] == 'half_cheetah_multi_vel':
        env = HalfCheetahMixtureEnvVel()
    elif config['env'] == 'hopper_multi':
        env = HopperMulti(config)
    elif config['env'] == 'walker_multi':
        env = WalkerMulti(config)
    elif config['env'] == 'ant_multi':
        env = AntMulti()
    elif config['env'] == 'half_cheetah_follow_toy':
        simple_env = Toy1dMultiTask()
        env = HalfCheetahMixtureToyEnv(config, simple_env)
    else:
        print('Invalid argument for environment name... Instantiating hopper env instead')
        env = HopperGoal()
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]
    epochs = config['epochs']
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
                task_dim = config['task_dim'],
                hidden_layers_actor = config['hidden_layers_actor'],
                hidden_layers_critic = config['hidden_layers_critic'],
                memory_size=config['memory_size'],
                batch_size=config['batch_size'],
                gamma=config['gamma'],
                alpha=config['alpha'],
                lr=config['lr'],
                action_bounds=action_bounds,
                reward_scale=reward_scale,
                pretrained=config['pretrained'])
    train(env, agent, epochs, experiment_name, save_after_episodes, policy_update_steps=512)