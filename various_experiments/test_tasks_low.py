from tigr.task_inference.dpmm_inference import DecoupledEncoder
# from configs.toy_config import toy_config
import numpy as np
from rlkit.envs import ENVS
from tigr.task_inference.dpmm_bnp import BNPModel
import torch
import os
from rlkit.torch.sac.policies import TanhGaussianPolicy
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from model import PolicyNetwork as TransferFunction
import rlkit.torch.pytorch_util as ptu
from collections import OrderedDict
import cv2
from typing import List, Any, Dict, Callable
import json
import imageio
import rlkit.torch.pytorch_util as ptu
from tigr.task_inference.prediction_networks import DecoderMDP, ExtendedDecoderMDP
import matplotlib.pyplot as plt
import random
from collections import namedtuple
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from sac_envs.walker import WalkerGoal
from sac_envs.hopper import HopperGoal
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from sac_envs.hopper_multi import HopperMulti
from sac_envs.walker_multi import WalkerMulti
from sac_envs.ant_multi import AntMulti
from sac_envs.walker_multi import WalkerMulti

from agent import SAC
from model import ValueNetwork, QvalueNetwork, PolicyNetwork

from mrl_analysis.plots.plot_settings import *

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = 'cuda'
if DEVICE=='cuda':
    ptu.set_gpu_mode(True)
else:
    ptu.set_gpu_mode(False)
    

experiment_name = 'retrain_hopper'
# TODO: einheitliches set to device
simple_env_dt = 0.05
sim_time_steps = 20
max_path_len=100
save_after_episodes = 10
plot_every = 5
batch_size = 20
policy_update_steps = 512

def log_all(agent, path, q1_loss, policy_loss, rew, episode, save_network='high_level', additional_plot=False):
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
    def save_plot(loss_history, name:str, path=f'{os.getcwd()}/experiment_plots', figure_size: Tuple[int,int] = (20, 10)):
        def remove_outliers_iqr(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = np.array(data)
            return data[(data >= lower_bound) & (data <= upper_bound)]

        def moving_average(data, window_size=10):
            index_array = np.arange(1, len(data) + 1)
            data = pd.Series(data, index=index_array)
            return data.rolling(window=window_size).mean()

        def format_label(label):
            words = label.split('_')
            return ' '.join(word.capitalize() for word in words)

        os.makedirs(path, exist_ok=True)
        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, np.arange(len(loss_history)), loss_history, label=format_label(name))
        axs.legend(fontsize=24)
        axs.set_xlabel('Train epochs', fontsize=20)
        axs.tick_params(axis='both', which='major', labelsize=20)
        axs.tick_params(axis='both', which='minor', labelsize=20)

        fig.savefig(os.path.join(path, name + '.png'))
        fig.savefig(os.path.join(path, name + '.pdf'))
        plt.close()
    
    path = os.path.join(path, experiment_name)
    # Save networks
    curr_path = path + '/models/policy_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'{save_network}.pth'
    if episode % 50 == 0:
        torch.save(agent.policy_network.cpu(), save_path)
    curr_path = path + '/models/vf1_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'{save_network}.pth'
    if episode % 50 == 0:
        torch.save(agent.q_value_network1.cpu(), save_path)
    curr_path = path + '/models/vf2_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'{save_network}.pth'
    if episode % 50 == 0:
        torch.save(agent.q_value_network2.cpu(), save_path)
    curr_path = path + '/models/value_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'{save_network}.pth'
    if episode % 50 == 0:
        torch.save(agent.value_network.cpu(), save_path)
    agent.q_value_network1.cuda() 
    agent.q_value_network2.cuda()
    agent.value_network.cuda()
    agent.policy_network.cuda() 

    # Save plots
    path_plots = os.path.join(path, 'plots', save_network)
    save_plot(q1_loss, name='vf_loss', path=path_plots)
    save_plot(rew, name='reward_history', path=path_plots)
    save_plot(policy_loss, name='policy_loss', path=path_plots)
    if additional_plot:
        save_plot(additional_plot['data'], name=additional_plot['name'], path=path_plots)

def get_encoder(path, shared_dim, encoder_input_dim):
    path = os.path.join(path, 'weights')
    for filename in os.listdir(path):
        if filename.startswith('encoder'):
            name = os.path.join(path, filename)
    
    # Important: Gru and Conv only work with trajectory encoding
    if variant['algo_params']['encoder_type'] in ['gru'] and variant['algo_params']['encoding_mode'] != 'trajectory':
        print(f'\nInformation: Setting encoding mode to trajectory since encoder type '
              f'"{variant["algo_params"]["encoder_type"]}" doesn\'t work with '
              f'"{variant["algo_params"]["encoding_mode"]}"!\n')
        variant['algo_params']['encoding_mode'] = 'trajectory'
    elif variant['algo_params']['encoder_type'] in ['transformer', 'conv'] and variant['algo_params']['encoding_mode'] != 'transitionSharedY':
        print(f'\nInformation: Setting encoding mode to trajectory since encoder type '
              f'"{variant["algo_params"]["encoder_type"]}" doesn\'t work with '
              f'"{variant["algo_params"]["encoding_mode"]}"!\n')
        variant['algo_params']['encoding_mode'] = 'transitionSharedY'

    encoder = DecoupledEncoder(
        shared_dim,
        encoder_input_dim,
        num_classes = variant['reconstruction_params']['num_classes'],
        latent_dim = variant['algo_params']['latent_size'],
        time_steps = variant['algo_params']['time_steps'],
        encoding_mode=variant['algo_params']['encoding_mode'],
        timestep_combination=variant['algo_params']['timestep_combination'],
        encoder_type=variant['algo_params']['encoder_type'],
        bnp_model=bnp_model
    )
    encoder.load_state_dict(torch.load(name, map_location='cpu'))
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.to(DEVICE)
    return encoder


def get_complex_agent(env, low_control_path):
    pretrained = low_control_path
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]
    transfer_function = TransferFunction(
        n_states=n_states,
        n_actions=n_actions,
        action_bounds=action_bounds,
        pretrained=pretrained
        )
    transfer_function.to(DEVICE)
    return transfer_function

def cheetah_to_simple_env_map(
    # observations: torch.Tensor, 
    observations,
    next_observations: torch.Tensor):
    """
    Maps transitions from the cheetah environment
    to the discrete, one-dimensional goal environment.
    """

    ### little help: [0:3] gives elements in positions 0,1,2 
    simple_observations = np.zeros(obs_dim)
    simple_observations[...,0] = observations[...,-3]
    simple_observations[...,1] = observations[...,8]

    next_simple_observations = np.zeros(obs_dim)
    next_simple_observations[...,0] = next_observations[...,-3]
    next_simple_observations[...,1] = next_observations[...,8]

    return simple_observations, next_simple_observations

def multiple_cheetah_to_simple_env_map(obs, next_obs, env):
    simple_observations = np.zeros(obs_dim)
    simple_observations[...,0] = obs[...,-3]
    simple_observations[...,1] = obs[...,0]
    simple_observations[...,2] = obs[...,1]
    simple_observations[...,3] = obs[...,8]
    simple_observations[...,4] = obs[...,9]
    simple_observations[...,5] = obs[...,10]

    next_simple_observations = np.zeros(obs_dim)
    next_simple_observations[...,0] = env.sim.data.qpos[0]
    next_simple_observations[...,1] = env.sim.data.qpos[1]
    next_simple_observations[...,2] = env.sim.data.qpos[2]
    next_simple_observations[...,3] = env.sim.data.qvel[0]
    next_simple_observations[...,4] = env.sim.data.qvel[1]
    next_simple_observations[...,5] = env.sim.data.qvel[2]

    return simple_observations, next_simple_observations


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
            # Apply transformation if any 
            if transform is not None:
                frame = transform(frame)
            else:
                # Convert color space if no transformation provided
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            writer.append_data(frame)

def get_decoder(path, action_dim, obs_dim, reward_dim, latent_dim, output_action_dim, net_complex_enc_dec, variant):
    path = os.path.join(path, 'weights')
    for filename in os.listdir(path):
        if filename.startswith('decoder'):
            name = os.path.join(path, filename)
    output_action_dim = 8
    decoder = ExtendedDecoderMDP(
        action_dim,
        obs_dim,
        reward_dim,
        latent_dim,
        output_action_dim,
        net_complex_enc_dec,
        variant['env_params']['state_reconstruction_clip'],
    ) 

    decoder.load_state_dict(torch.load(name, map_location='cpu'))
    decoder.to(DEVICE)
    return decoder

def rollout(env, encoder, decoder, high_level_controller, step_predictor, transfer_function, 
            variant, obs_dim, actions_dim, max_path_len, 
            n_tasks, inner_loop_steps, save_video_path, max_steps=sim_time_steps, beta = 0.05, batch_size=batch_size, policy_update_steps=policy_update_steps):
    range_dict = OrderedDict(pos_x = [0.5, 25],
                             velocity_z = [1.5, 3.],
                             pos_y = [np.pi / 6., np.pi / 2.],
                             velocity_x = [0.5, 5.0],
                             velocity_y = [2. * np.pi, 4. * np.pi],
                             )
    
    value_loss_history, q_loss_history, policy_loss_history, rew_history = [], [], [], []
    low_value_loss_history, low_q_loss_history, low_policy_loss_history, low_rew_history, step_history = [], [], [], [], []
    path = save_video_path
    
    with open(f'{save_video_path}/weights/stats_dict.json', 'r') as file:
        stats_dict = json.load(file)

    
    x_pos_plot = []
    x_vel_plot = []
    jump_plot = []
    rot_plot = []
    tasks_vel = []
    tasks_pos = []
    tasks_jump = []
    tasks_rot =[]
    for episode in range(len(tasks)):

        
        
            
        print("\033[1m" + f"Episode: {episode}" + "\033[0m")
        print(experiment_name)

        l_vars = []
        labels = []

        done = 0
        print('Rolling out ...')
        plot = False
        record_video = False
        frames = []
        image_info = dict(reward = [],
        obs = [],
        base_task = [],
        action = [])
        plot = True
        record_video = False



        episode_reward = 0
        low_episode_reward = 0
        rew, low_rew = [], []
        x_pos_curr, x_vel_curr, jump_curr, rot_curr = [],[], [], []
        path_length = 0
        obs = env.reset()[0]
        x_pos_curr.append(env.sim.data.qpos[0])
        x_vel_curr.append(env.sim.data.qvel[0])
        jump_curr.append(np.abs(env.sim.data.qvel[1]))
        rot_curr.append(env.sim.data.qpos[2])
        contexts = torch.zeros((n_tasks, variant['algo_params']['time_steps'], obs_dim + 1 + obs_dim), device=DEVICE)
        task = env.sample_task(test=True, task=tasks[episode])
        # count=0
        # while env.base_task not in [
        #                             env.config.get('tasks',{}).get('goal_front'), 
        #                             env.config.get('tasks',{}).get('goal_back'),
        #                             env.config.get('tasks',{}).get('forward_vel'), 
        #                             env.config.get('tasks',{}).get('backward_vel')]:
        #     task = env.sample_task(test=True)
        #     count+=1
        #     if count>150:
        #         return 'Failed to sample task. Attempted 150 times.'
        # env.update_task(task)

        sim_time_steps_list = []

        # if env.base_task in [env.config.get('tasks',{}).get('goal_front'), env.config.get('tasks',{}).get('goal_back')]:
        #     max_path_len = 300
        while path_length < max_path_len:

            # get encodings
            encoder_input = contexts.detach().clone()
            encoder_input = encoder_input.view(encoder_input.shape[0], -1).to(DEVICE)
            mu, log_var = encoder(encoder_input)     # Is this correct??
            obs_before_sim = env._get_obs()

            # Save latent vars
            # policy_input = torch.cat([ptu.from_numpy(obs), mu.squeeze()], dim=-1)
            action_prev = high_level_controller.choose_action(obs.squeeze(), mu.cpu().detach().numpy().squeeze(), torch=True, max_action=False, sigmoid=True).squeeze()
            # action_prev[env.config.get('tasks',{}).get('stand_back')] = 0

            # TODO: Not best suited for position since once it reaches the position, it must stay at that position. Think about an alternative
            base_task_pred = torch.argmax(torch.abs(action_prev))
            # action_prev[[1,2,4]] = 0 * action_prev[[1,2,4]]
            # action = action_prev
            #TODO: redo this with new trainings
            action = torch.zeros_like(action_prev).to(DEVICE)
            action[base_task_pred] = action_prev[base_task_pred]
            desired_state = torch.zeros_like(action).to(DEVICE)
            if base_task_pred == env.config.get('tasks',{}).get('goal_front'):
                action_normalize = action[base_task_pred].item()
                action[base_task_pred] = action[base_task_pred] + env.sim.data.qpos[0]
                #TODO: change to desired_state[base_task_pred] = action[base_task_pred], do a function that maps the base task to desired state
                desired_state[base_task_pred] = action[base_task_pred]
            elif base_task_pred == env.config.get('tasks',{}).get('goal_back'):
                action_normalize = action[base_task_pred].item()
                action[base_task_pred] = - action[base_task_pred] + env.sim.data.qpos[0]
                #TODO: change to desired_state[base_task_pred] = action[base_task_pred], do a function that maps the base task to desired state
                desired_state[base_task_pred] = action[base_task_pred]
            elif base_task_pred == env.config.get('tasks',{}).get('forward_vel'):
                action[base_task_pred] = action[base_task_pred] * range_dict['velocity_x'][1]
                desired_state[base_task_pred] = action[base_task_pred]
            elif base_task_pred == env.config.get('tasks',{}).get('backward_vel'):
                action[base_task_pred] = - action[base_task_pred] * range_dict['velocity_x'][1]
                desired_state[base_task_pred] = action[base_task_pred]
            elif base_task_pred == env.config.get('tasks',{}).get('jump'):
                action[base_task_pred] = action[base_task_pred] * range_dict['velocity_z'][1]
                desired_state[base_task_pred] = action[base_task_pred]
            elif base_task_pred == env.config.get('tasks',{}).get('stand_front'):
                action[base_task_pred] = action[base_task_pred] * range_dict['pos_y'][1]
                desired_state[base_task_pred] = action[base_task_pred]
            elif base_task_pred == env.config.get('tasks',{}).get('stand_back'):
                action[base_task_pred] = - action[base_task_pred] * range_dict['pos_y'][1]
                desired_state[base_task_pred] = action[base_task_pred]


            r = 0
            sim_time_steps = int(torch.clamp(step_predictor.choose_action(obs, desired_state.detach().cpu().numpy(), torch=True).squeeze()*max_steps,1,max_steps))
            for i in range(sim_time_steps):
                complex_action = transfer_function.get_action(ptu.from_numpy(obs), action, return_dist=False)
                # complex_action = transfer_function.get_action(ptu.from_numpy(obs), action, return_dist=False)
                next_obs, step_r, done, truncated, env_info = env.step(complex_action.detach().cpu().numpy(), healthy_scale=0)

                penalty_r = action_prev
                # penalty_r[base_task_pred] = 0
                # step_r -= 0.05 * torch.sum(torch.square(penalty_r)).detach().cpu().numpy()
                step_r -= 0.05 * torch.sum(torch.abs(penalty_r)).detach().cpu().numpy()
                # step_r = step_r.clip(-3, 0)
                obs = next_obs
                if record_video:
                    image = env.render()
                    frames.append(image)
                    image_info['reward'].append(r)
                    image_info['obs'].append(cheetah_to_simple_env_map(obs_before_sim, obs)[1])
                    image_info['base_task'].append(env.task)
                    image_info['action'].append(action)
                r = step_r
            x_pos_curr.append(env.sim.data.qpos[0])
            x_vel_curr.append(env.sim.data.qvel[0])
            jump_curr.append(np.abs(env.sim.data.qvel[1]))
            rot_curr.append(env.sim.data.qpos[2])

            path_length+=1

            # r = r/sim_time_steps
            sim_time_steps_list.append(sim_time_steps)

            if base_task_pred in [env.config.get('tasks',{}).get('goal_front')]:
                low_level_r = - np.abs(action[env.config['tasks']['goal_front']].detach().cpu().numpy()-env.sim.data.qpos[0])/np.abs(action_normalize) - beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('goal_back')]:
                low_level_r = - np.abs(action[env.config['tasks']['goal_back']].detach().cpu().numpy()-env.sim.data.qpos[0])/np.abs(action_normalize) - beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('forward_vel')]:
                low_level_r = - np.abs(action[env.config['tasks']['forward_vel']].detach().cpu().numpy()-env.sim.data.qvel[0])/np.abs(action[env.config['tasks']['forward_vel']].item()) - beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('backward_vel')]:
                low_level_r = - np.abs(action[env.config['tasks']['backward_vel']].detach().cpu().numpy()-env.sim.data.qvel[0])/np.abs(action[env.config['tasks']['backward_vel']].item()) - beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('stand_front')]:
                low_level_r = - np.abs(action[env.config['tasks']['stand_front']].detach().cpu().numpy()-env.sim.data.qpos[2])/np.abs(action[env.config['tasks']['stand_front']].item()) - beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('stand_back')]:
                low_level_r = - np.abs(action[env.config['tasks']['stand_back']].detach().cpu().numpy()-env.sim.data.qpos[2])/np.abs(action[env.config['tasks']['stand_back']].item()) - beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('jump')]:
                low_level_r = - np.abs(action[env.config['tasks']['jump']].detach().cpu().numpy()-np.abs(env.sim.data.qvel[1]))/np.abs(action[env.config['tasks']['jump']].item()) - beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)

            episode_reward += r
            low_episode_reward += low_level_r
            
            prev_simple_obs, next_simple_obs = cheetah_to_simple_env_map(obs_before_sim, obs)
            
            data = torch.cat([ptu.from_numpy(prev_simple_obs), torch.unsqueeze(torch.tensor(r, device=DEVICE), dim=0), ptu.from_numpy(next_simple_obs)], dim=0).unsqueeze(dim=0)
            context = torch.cat([contexts.squeeze(), data], dim=0)
            contexts = context[-time_steps:, :]
            contexts = contexts.unsqueeze(0).to(torch.float32)


        if env.base_task in [env.config.get('tasks',{}).get('goal_front'), env.config.get('tasks',{}).get('goal_back')]:
            tasks_pos.append(task[env.base_task])
            x_pos_plot.append(np.array(x_pos_curr))
        elif env.base_task in [env.config.get('tasks',{}).get('forward_vel'), env.config.get('tasks',{}).get('backward_vel')]:
            x_vel_plot.append(np.array(x_vel_curr))
            tasks_vel.append(task[env.base_task])
        elif env.base_task in [env.config.get('tasks',{}).get('stand_front'), env.config.get('tasks',{}).get('stand_back')]:
            rot_plot.append(np.array(rot_curr))
            tasks_rot.append(task[env.base_task])
        elif env.base_task == env.config.get('tasks',{}).get('jump'):
            jump_plot.append(np.array(jump_curr))
            tasks_jump.append(task[env.base_task])

        if plot:
            # size = frames[0].shape
            window_size = 10

            def moving_average(data, window_size):
                # """Compute the moving average using a sliding window."""
                # window = np.ones(int(window_size))/float(window_size)
                # return np.convolve(data, window, 'same')
                from scipy.ndimage.filters import gaussian_filter1d
                return gaussian_filter1d(data, sigma=2)
            
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            ax1, ax2 = ax.flatten()
            fig.subplots_adjust(hspace=0.4, wspace=0.4)


            for i, x_pos in enumerate(x_pos_plot):
                color = f'C{i}'
                x_pos = moving_average(x_pos, window_size)
                # Plot position on the first (left) axis
                ax1.plot(np.arange(len(x_pos)), np.array(x_pos), label='Position', color=color)
                # if tasks[i][0]!=0:
                ax1.plot(np.arange(len(x_pos)), np.ones(len(x_pos))*tasks_pos[i], linestyle='--', color=color, alpha=0.5)
            
            ax1.set_xlabel('Time (s)', fontsize=32)
            ax1.set_ylabel('Position (m)', fontsize=32)
            ax1.tick_params(axis='y', labelsize=24)
            ax1.tick_params(axis='x', labelsize=24)
            # Create a second axis sharing the same x-axis
            for i, x_vel in enumerate(x_vel_plot):
                color = f'C{i}'
                x_vel = moving_average(x_vel, window_size)
                ax2.plot(np.arange(len(x_vel)), np.array(x_vel), label='Velocity', color=color)
                # if tasks[i][3]!=0:
                ax2.plot(np.arange(len(x_vel)), np.ones(len(x_vel))*tasks_vel[i], linestyle='--', color=color, alpha=0.5)
            ax2.set_xlabel('Time (s)', fontsize=32)
            ax2.tick_params(axis='y', labelsize=24)
            ax2.tick_params(axis='x', labelsize=24)
            ax2.set_ylabel('Velocity (m/s)', fontsize=32)
            # ax2.set_title(f'Avg Reward: {vel_reward/(episode+1)}')
            plt.tight_layout()

            # for i, jump in enumerate(jump_plot):
            #     color = f'C{i}'
            #     jump = moving_average(jump, window_size)
            #     # Plot position on the first (left) axis
            #     ax3.plot(np.arange(len(jump)), np.array(jump), label='Jump', color=color)
            #     # if tasks[i][0]!=0:
            #     ax3.plot(np.arange(len(jump)), np.ones(len(jump))*tasks_jump[i], linestyle='--', color=color, alpha=0.5)
            
            # ax3.set_xlabel('Time (s)')
            # ax3.set_ylabel('Jump (m/s)')
            # ax3.tick_params(axis='y')

            # for i, rot in enumerate(rot_plot):
            #     color = f'C{i}'
            #     rot = moving_average(rot, window_size)
            #     # Plot position on the first (left) axis
            #     ax4.plot(np.arange(len(rot)), np.array(rot), label='Rot', color=color)
            #     # if tasks[i][0]!=0:
            #     ax4.plot(np.arange(len(rot)), np.ones(len(rot))*tasks_rot[i], linestyle='--', color=color, alpha=0.5)
            
            # ax4.set_xlabel('Time (s)')
            # ax4.set_ylabel('Rot(m)')
            # ax4.tick_params(axis='y')

            # Save the figure to a file
            dir = Path(os.path.join(save_video_path, experiment_name, 'trajectories_plots'))
            filename = os.path.join(dir,f"final.png")
            filename = os.path.join(dir,f"final.pdf")
            dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(filename, dpi=300)  # Save the figure with 300 dpi resolution
            plt.close()

        if record_video:
            save_as = f'{save_video_path}/videos/{experiment_name}/transfer_{episode}.mp4'
            _frames_to_gif(frames, image_info, save_as)


        low_rew.append(low_episode_reward)

        rew.append(episode_reward)

        



    return l_vars, labels
        

if __name__ == "__main__":
    # from experiments_configs.half_cheetah_multi_env import config as env_config

    # List of inference paths to test
    inference_paths = [
        {'name': 'var_0.1', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_11_10_26_00_default_true_gmm'},
        # {'name': 'no_var', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_02_15_23_09_default_true_gmm'},
        # {'name': 'no random step 10', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_11_11_49_42_default_true_gmm'},
        # {'name': 'var_0.2', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_12_09_17_42_default_true_gmm'},
        # {'name': 'var_0.01', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_12_20_31_24_default_true_gmm'},
        # {'name': 'var_0.1, step 10', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_14_20_55_42_default_true_gmm'},
        # {'name': 'var_0.1.2, step 10', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_12_23_15_default_true_gmm'},
        # {'name': 'var_0.1.3, step 10', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_15_34_37_default_true_gmm'},
        # {'name': 'inference_path_3', 'path': '/path/to/inference_path_3'},
    ]

    # You may need to adjust these paths and configurations as per your setup
    # complex_agent_config = dict(
    #     environment = HalfCheetahMixtureEnv,
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'new_cheetah_training/half_cheetah_initial_random',
    #     epoch = 700,
    # )
    complex_agent_config = dict(
        environment = HopperMulti,
        experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
        experiment_name = 'hopper_full_sac0.2_reward1_randomchange',
        epoch = 1400,
    )
    # complex_agent_config = dict(
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'walker_full_06_07',
    #     epoch = 2100,
    # )

    # Initialize rewards data
    rewards_data = {}

    # Loop over inference paths
    for inference in inference_paths:
        current_inference_path_name = inference['name']
        inference_path = inference['path']
        rewards_data[current_inference_path_name] = {'velocity': [], 'position': []}

        

        with open(complex_agent_config['experiments_repo'] + complex_agent_config['experiment_name'] + '/config.json', 'r') as file:
            env_config = json.load(file)

        if env_config['env'] == 'hopper':
            env = HopperGoal()
        elif env_config['env'] == 'walker':
            env = WalkerGoal()
        elif env_config['env'] == 'half_cheetah_multi':
            env = HalfCheetahMixtureEnv(env_config)
        elif env_config['env'] == 'hopper_multi':
            env = HopperMulti(env_config)
        elif env_config['env'] == 'walker_multi':
            env = WalkerMulti(env_config)
        elif env_config['env'] == 'ant_multi':
            env = AntMulti()
        env.render_mode = 'rgb_array'

        with open(f'{inference_path}/variant.json', 'r') as file:
            variant = json.load(file)

        # ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])

        m = variant['algo_params']['sac_layer_size']
        simple_env = ENVS[variant['env_name']](**variant['env_params'])         # Just used for initilization purposes

        ### PARAMETERS ###
        obs_dim = int(np.prod(simple_env.observation_space.shape))
        action_dim = int(np.prod(simple_env.action_space.shape))
        net_complex_enc_dec = variant['reconstruction_params']['net_complex_enc_dec']
        latent_dim = variant['algo_params']['latent_size']
        time_steps = variant['algo_params']['time_steps']
        num_classes = variant['reconstruction_params']['num_classes']
        # max_path_len = variant['algo_params']['max_path_length']
        reward_dim = 1
        encoder_input_dim = time_steps * (obs_dim + reward_dim + obs_dim)
        shared_dim = int(encoder_input_dim / time_steps * net_complex_enc_dec)
        if variant['algo_params']['sac_context_type']  == 'sample':
            policy_latent_dim = latent_dim
        else:
            policy_latent_dim  = latent_dim * 2

        
        bnp_model = BNPModel(
            save_dir=variant['dpmm_params']['save_dir'],
            start_epoch=variant['dpmm_params']['start_epoch'],
            gamma0=variant['dpmm_params']['gamma0'],
            num_lap=variant['dpmm_params']['num_lap'],
            fit_interval=variant['dpmm_params']['fit_interval'],
            kl_method=variant['dpmm_params']['kl_method'],
            birth_kwargs=variant['dpmm_params']['birth_kwargs'],
            merge_kwargs=variant['dpmm_params']['merge_kwargs']
        )
        pretrained_policy = f'{inference["path"]}/{experiment_name}'
        high_level_controller = SAC(n_states=env.observation_space.shape[0],
                    n_actions=env.action_space.shape[0],
                    task_dim = variant['algo_params']['latent_size'],
                    out_actions = env.task.shape[0],
                    hidden_layers_actor = [300,300,300,300],
                    hidden_layers_critic = [300,300,300,300],
                    memory_size=1e+6,
                    batch_size=256,
                    gamma=0.9,
                    alpha=0.2,
                    lr=3e-4,
                    action_bounds=[-50,50],
                    reward_scale=5, 
                    pretrained=dict(path=pretrained_policy, file_name='high_level')
                    )
        output_action_dim = 8
        # decoder = get_decoder(inference_path, action_dim, obs_dim, reward_dim, latent_dim, output_action_dim, net_complex_enc_dec, variant)
        decoder = None
        step_predictor = SAC(n_states=env.observation_space.shape[0],
                    n_actions=1,
                    task_dim = env.task.shape[0],   # desired state
                    hidden_layers_actor = [64,64,64,64,64],
                    hidden_layers_critic = [64,64,64,64,64],
                    memory_size=1e+6,
                    batch_size=512,
                    gamma=0.9,
                    alpha=0.2,
                    lr=3e-4,
                    action_bounds=[-50,50],
                    reward_scale=1, 
                    pretrained=dict(path=pretrained_policy, file_name='low_level')
                    )

        encoder = get_encoder(inference_path, shared_dim, encoder_input_dim)
        transfer_function = get_complex_agent(env, f'{inference_path}/{experiment_name}/models/policy_model/transfer_function.pth')
        output_action_dim = env.task.shape[0]
        decoder = get_decoder(inference_path, action_dim, obs_dim, reward_dim, latent_dim, output_action_dim, net_complex_enc_dec, variant)
        optimizer = optim.Adam(decoder.parameters(), lr=3e-4)

        tasks = [
            {'base_task':'goal_back', 'specification':1},
            {'base_task':'goal_back', 'specification':0.66},
            # {'base_task':'goal_back', 'specification':0.5},
            {'base_task':'goal_back', 'specification':0.33},
            # {'base_task':'goal_front', 'specification':0.25},
            {'base_task':'goal_front', 'specification':0.33},
            {'base_task':'goal_front', 'specification':0.66},
            {'base_task':'goal_front', 'specification':1},
            {'base_task':'backward_vel', 'specification':0.8},
            {'base_task':'backward_vel', 'specification':0.5},
            {'base_task':'backward_vel', 'specification':0.1},
            # {'base_task':'backward_vel', 'specification':1.0},
            # {'base_task':'forward_vel', 'specification':1.0},
            {'base_task':'forward_vel', 'specification':0.1},
            {'base_task':'forward_vel', 'specification':0.5},
            {'base_task':'forward_vel', 'specification':0.8},
                 ]
        rollout(env, encoder, decoder, high_level_controller, step_predictor,
                                        transfer_function, variant, obs_dim, action_dim, 
                                        max_path_len, n_tasks=1, inner_loop_steps=10, save_video_path=inference_path)

    # After all rollouts are complete, create the box plots
    import matplotlib.pyplot as plt

    # Prepare data for box plots
    boxplot_data = []
    x_labels = []
    positions = []
    pos = 1  # Starting position for the first box

    for i, inference in enumerate(inference_paths):
        inference_name = inference['name']
        # Get rewards for position and velocity tasks
        position_rewards = rewards_data[inference_name]['position']
        velocity_rewards = rewards_data[inference_name]['velocity']
        
        # Append data
        boxplot_data.extend([position_rewards, velocity_rewards])
        
        # Append labels
        x_labels.extend([f"{inference_name}\nPosition", f"{inference_name}\nVelocity"])
        
        # Append positions
        positions.extend([pos, pos + 1])
        
        # Update position for next inference path
        pos += 3  # Adding space between groups]