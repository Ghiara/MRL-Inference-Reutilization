from tigr.task_inference.dpmm_inference import DecoupledEncoder
# from configs.toy_config import toy_config
import numpy as np
from rlkit.envs import ENVS
from tigr.task_inference.dpmm_bnp import BNPModel
import torch
import os
from rlkit.torch.sac.policies import TanhGaussianPolicy
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from sac_envs.hopper_multi import HopperMulti
from sac_envs.walker_multi import WalkerMulti
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
from various_experiments.replay_memory import Memory
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Tuple

from agent import SAC
from model import ValueNetwork, QvalueNetwork, PolicyNetwork
from mrl_analysis.utility.data_smoothing import smooth_plot, smooth_fill_between
from vis_utils.logging import log_all, _frames_to_gif

from mrl_analysis.plots.plot_settings import *

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = 'cuda'
ptu.set_gpu_mode(True)

experiment_name = 'retrain_hopper'
# TODO: einheitliches set to device
simple_env_dt = 0.05
sim_max_time_steps = 20
max_path_len=100
save_after_episodes = 5
plot_every = 5
batch_size = 10
policy_update_steps = 1024

def get_encoder(path, shared_dim, encoder_input_dim):

    '''
    This function is used to load the encoder trained on the toy agent given by path
    '''

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


def get_complex_agent(env, complex_agent_config, train=False):

    '''
    This function is used to load the low-level controller specifide by comlpex_agent_config
    '''

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]
    
    if train:
        pretrained = dict(path=os.path.join(complex_agent_config['experiments_repo'], complex_agent_config['experiment_name']),
                               epoch = complex_agent_config['epoch'])
        transfer_function = SAC(n_states=n_states,
                n_actions=n_actions,
                task_dim = env.task.shape[0],
                hidden_layers_actor = [300,300,300],
                hidden_layers_critic = [300,300,300],
                memory_size=1e+6,
                batch_size=256,
                gamma=0.99,
                alpha=0.2,
                lr=3e-5,
                action_bounds=[-50,50],
                reward_scale=5, pretrained=pretrained)
        
    else:
        pretrained = complex_agent_config['experiments_repo']+complex_agent_config['experiment_name']+f"/models/policy_model/epoch_{complex_agent_config['epoch']}.pth"
        transfer_function = TransferFunction(
            n_states=n_states,
            n_actions=n_actions,
            action_bounds=action_bounds,
            pretrained=pretrained
            )
        transfer_function.to(DEVICE)
        
    return transfer_function

def cheetah_to_simple_env_map(
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

def rollout(env, encoder, decoder, high_level_controller, step_predictor, transfer_function, 
            variant, obs_dim, actions_dim, max_path_len, 
            n_tasks, inner_loop_steps, save_video_path, max_steps=sim_max_time_steps, beta = 0.1, batch_size=batch_size, policy_update_steps=policy_update_steps):
    range_dict = OrderedDict(pos_x = [0.5, 25],
                             velocity_z = [1.5, 3.],
                             pos_y = [np.pi / 6., np.pi / 2.],
                             velocity_x = [0.5, 5.0],
                             velocity_y = [2. * np.pi, 4. * np.pi],
                             )
    
    value_loss_history, q_loss_history, policy_loss_history, rew_history = [], [], [], []
    low_value_loss_history, low_q_loss_history, low_policy_loss_history, low_rew_history, step_history = [], [], [], [], []
    transfer_function_rew_history, transfer_function_value_loss_history, transfer_function_policy_loss_history, transfer_function_q_loss_history = [], [], [], []
    path = save_video_path
    
    with open(f'{save_video_path}/weights/stats_dict.json', 'r') as file:
        stats_dict = json.load(file)

    
    x_pos_plot = []
    x_vel_plot = []
    tasks_vel = []
    tasks_pos = []
    for episode in range(30000):

        
        
            
        print("\033[1m" + f"Episode: {episode}" + "\033[0m")
        print(experiment_name)

        l_vars = []
        labels = []

        value_loss, q_loss, policy_loss = [], [], []
        low_value_loss, low_q_loss, low_policy_loss = [], [], []
        transfer_function_value_loss, transfer_function_policy_loss, transfer_function_q_loss = [],[],[]
        print('Collecting samples...')
        for batch in tqdm(range(batch_size)):
            plot = False
            if episode % plot_every == 0 and batch == 0:
                frames = []
                image_info = dict(reward = [],
                obs = [],
                base_task = [],
                action = [])
                plot = True

            
            done = 0
            episode_reward = 0
            low_episode_reward = 0
            rew, low_rew = [], []
            x_pos_curr, x_vel_curr = [],[]
            path_length = 0
            obs = env.reset()[0]
            x_pos_curr.append(env.sim.data.qpos[0])
            x_vel_curr.append(env.sim.data.qvel[0])
            contexts = torch.zeros((n_tasks, variant['algo_params']['time_steps'], obs_dim + 1 + obs_dim), device=DEVICE)
            task = env.sample_task(test=True)
            count = 0
            while env.base_task not in [env.config.get('tasks',{}).get('goal_front'), 
                                        env.config.get('tasks',{}).get('goal_back'),
                                        env.config.get('tasks',{}).get('forward_vel'), 
                                        env.config.get('tasks',{}).get('backward_vel')]:
                task = env.sample_task(test=True)
                count+=1
                if count>150:
                    return 'Failed to sample task. Attempted 150 times.'
            env.update_task(task)

            sim_time_steps_list = []

            while (not done) and path_length < max_path_len:
                path_length += 1
                # get encodings
                encoder_input = contexts.detach().clone()
                encoder_input = encoder_input.view(encoder_input.shape[0], -1).to(DEVICE)
                mu, log_var = encoder(encoder_input)

                obs_before_sim = env._get_obs()

                action_prev = high_level_controller.choose_action(obs.squeeze(), mu.cpu().detach().numpy().squeeze(), torch=True, max_action=False, sigmoid=True).squeeze()

                action = torch.zeros_like(action_prev).to(DEVICE)
                action[torch.argmax(torch.abs(action_prev))] = action_prev[torch.argmax(torch.abs(action_prev))]
                if action[env.config.get('tasks',{}).get('goal_front')] != 0:
                    action_normalize = action[env.config['tasks']['goal_front']].item()
                    action[env.config['tasks']['goal_front']] = action[env.config['tasks']['goal_front']] + env.sim.data.qpos[0]
                    desired_state = torch.tensor([action[env.config['tasks']['goal_front']], 0])
                elif action[env.config.get('tasks',{}).get('goal_back')] != 0:
                    action_normalize = action[env.config['tasks']['goal_back']].item()
                    action[env.config['tasks']['goal_back']] = - action[env.config['tasks']['goal_back']] + env.sim.data.qpos[0]
                    desired_state = torch.tensor([action[env.config['tasks']['goal_back']], 0])
                elif action[env.config.get('tasks',{}).get('forward_vel')] != 0:
                    action[env.config['tasks']['forward_vel']] = action[env.config['tasks']['forward_vel']] * range_dict['velocity_x'][1]
                    desired_state = torch.tensor([0, action[env.config['tasks']['forward_vel']]])
                elif action[env.config.get('tasks',{}).get('backward_vel')] != 0:
                    action[env.config['tasks']['backward_vel']] = - action[env.config['tasks']['backward_vel']] * range_dict['velocity_x'][1]
                    desired_state = torch.tensor([0, action[env.config['tasks']['backward_vel']]])

                r = 0
                sim_time_steps = int(torch.clamp(step_predictor.choose_action(obs, desired_state, torch=True).squeeze()*max_steps,1,max_steps))

                i = 0
                while (i < sim_time_steps) and (not done):
                    i+=1
                    complex_action = transfer_function.choose_action(obs, action.detach().cpu().numpy(), torch=True).squeeze()
                    next_obs, step_r, done, truncated, env_info = env.step(complex_action.detach().cpu().numpy(), healthy_scale=0)

                    simple_obs = [env.sim.data.qpos[0], env.sim.data.qvel[0]]
                    if not done:
                        healthy_reward = 1
                    else:
                        healthy_reward = 0
                    # healthy_reward = 0
                    if action[env.config.get('tasks',{}).get('goal_front')] != 0:
                        sim_steps_r = - np.abs(action[env.config['tasks']['goal_front']].detach().cpu().numpy()-simple_obs[0])/(np.abs(action_normalize)+0.5)
                        sim_steps_r = np.clip(sim_steps_r, -5, 1)
                    elif action[env.config.get('tasks',{}).get('goal_back')] != 0:
                        sim_steps_r = - np.abs(action[env.config['tasks']['goal_back']].detach().cpu().numpy()-simple_obs[0])/(np.abs(action_normalize)+0.5)
                        sim_steps_r = np.clip(sim_steps_r, -5, 1)
                    elif action[env.config.get('tasks',{}).get('forward_vel')] != 0:
                        sim_steps_r = - np.abs(action[env.config['tasks']['forward_vel']].detach().cpu().numpy()-simple_obs[1])/(np.abs(action[env.config['tasks']['forward_vel']].item())+0.5)
                        sim_steps_r = np.clip(sim_steps_r, -5, 1)
                    elif action[env.config.get('tasks',{}).get('backward_vel')] != 0:
                        sim_steps_r = - np.abs(action[env.config['tasks']['backward_vel']].detach().cpu().numpy()-simple_obs[1])/(np.abs(action[env.config['tasks']['backward_vel']].item())+0.5)
                        sim_steps_r = np.clip(sim_steps_r, -5, 1)

                    low_level_r = sim_steps_r + healthy_reward
                    transfer_function.store(obs, low_level_r, done, complex_action.cpu().detach().numpy().squeeze(), next_obs, action)
                    transfer_function_losses = transfer_function.train(episode, False)
                    transfer_function_value_loss.append(transfer_function_losses[0])
                    transfer_function_policy_loss.append(transfer_function_losses[2])
                    transfer_function_q_loss.append(transfer_function_losses[1])
                    
                    step_r -= 0.05 * torch.sum(torch.square(action_prev)).detach().cpu().numpy()
                    step_r = step_r.clip(-2, 2)
                    obs = next_obs
                    if plot:
                        image = env.render()
                        frames.append(image)
                        image_info['reward'].append(r)
                        image_info['obs'].append(cheetah_to_simple_env_map(obs_before_sim, obs)[0])
                        image_info['base_task'].append(env.task)
                        image_info['action'].append(action)
                    r = step_r
                    low_episode_reward += low_level_r

                sim_time_steps_list.append(sim_time_steps)

                sim_steps_r -= beta * sim_time_steps
                step_predictor.store(obs_before_sim, sim_steps_r, done, np.array([sim_time_steps]), next_obs, desired_state)

                episode_reward += r
                low_level_r = low_level_r / sim_time_steps

                high_level_controller.store(obs_before_sim, r, done, action.cpu().detach().numpy().squeeze(), obs, mu.detach())
                
                prev_simple_obs, next_simple_obs = cheetah_to_simple_env_map(obs_before_sim, obs)
                
                data = torch.cat([ptu.from_numpy(prev_simple_obs), torch.unsqueeze(torch.tensor(r, device=DEVICE), dim=0), ptu.from_numpy(next_simple_obs)], dim=0).unsqueeze(dim=0)
                context = torch.cat([contexts.squeeze(), data], dim=0)
                contexts = context[-time_steps:, :]
                contexts = contexts.unsqueeze(0).to(torch.float32)

                
                

                x_pos_curr.append(env.sim.data.qpos[0])
                x_vel_curr.append(env.sim.data.qvel[0])

            if env.base_task in [env.config.get('tasks',{}).get('goal_front'), env.config.get('tasks',{}).get('goal_back')]:
                tasks_pos.append(task[env.base_task])
                x_pos_plot.append(np.array(x_pos_curr))
            elif env.base_task in [env.config.get('tasks',{}).get('forward_vel'), env.config.get('tasks',{}).get('backward_vel')]:
                x_vel_plot.append(np.array(x_vel_curr))
                tasks_vel.append(task[env.base_task])
            if len(x_pos_plot) > 5:
                x_pos_plot.pop(0)
                tasks_pos.pop(0)
            if len(x_vel_plot) > 5:
                x_vel_plot.pop(0)
                tasks_vel.pop(0)

            if plot and len(tasks_pos)>1 and len(tasks_pos)>1:
                size = frames[0].shape
                window_size = 10

                def moving_average(data, window_size):
                    # """Compute the moving average using a sliding window."""
                    # window = np.ones(int(window_size))/float(window_size)
                    # return np.convolve(data, window, 'same')
                    from scipy.ndimage.filters import gaussian_filter1d
                    return gaussian_filter1d(data, sigma=2)
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
                colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']


                for i, x_pos in enumerate(x_pos_plot):
                    color = f'C{i}'
                    x_pos = moving_average(x_pos, window_size)
                    # Plot position on the first (left) axis
                    ax1.plot(np.arange(len(x_pos)), np.array(x_pos), label='Position', color=color)
                    # if tasks[i][0]!=0:
                    ax1.plot(np.arange(len(x_pos)), np.ones(len(x_pos))*tasks_pos[i], linestyle='--', color=color, alpha=0.5)
                
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Position (m)')
                ax1.tick_params(axis='y')

                # Create a second axis sharing the same x-axis
                for i, x_vel in enumerate(x_vel_plot):
                    color = f'C{i}'
                    x_vel = moving_average(x_vel, window_size)
                    ax2.plot(np.arange(len(x_vel)), np.array(x_vel), label='Velocity', color=color)
                    # if tasks[i][3]!=0:
                    ax2.plot(np.arange(len(x_vel)), np.ones(len(x_vel))*tasks_vel[i], linestyle='--', color=color, alpha=0.5)
                ax2.set_xlabel('Time (s)')
                ax2.tick_params(axis='y')
                ax2.set_ylabel('Velocity (m/s)')

                # Save the figure to a file
                dir = Path(os.path.join(save_video_path, experiment_name, 'trajectories_plots'))
                filename = os.path.join(dir,f"epoch_{episode}.png")
                filename = os.path.join(dir,f"epoch_{episode}.pdf")
                dir.mkdir(exist_ok=True, parents=True)
                plt.savefig(filename, dpi=300)  # Save the figure with 300 dpi resolution
                plt.close()

                save_as = f'{save_video_path}/videos/{experiment_name}/transfer_{episode}.mp4'
                _frames_to_gif(frames, image_info, save_as)


            low_rew.append(low_episode_reward)

            rew.append(episode_reward)
        print('Training policies...')
        for k in tqdm(range(policy_update_steps)):

            # Train step predictor
            low_level_losses = step_predictor.train(episode, False)
            low_value_loss.append(low_level_losses[0])
            low_policy_loss.append(low_level_losses[2])
            low_q_loss.append(low_level_losses[1])

            # Train high_level controller
            losses = high_level_controller.train(episode, False)
            value_loss.append(losses[0])
            policy_loss.append(losses[2])
            q_loss.append(losses[1]) 

            # # Train transfer function
            # transfer_function_losses = transfer_function.train(episode, False)
            # transfer_function_value_loss.append(transfer_function_losses[0])
            # transfer_function_policy_loss.append(transfer_function_losses[2])
            # transfer_function_q_loss.append(transfer_function_losses[1])

        print('Task:', task[env.base_task], 'Base task:', env.base_task)
        print('END obs:', env.sim.data.qpos[0], env.sim.data.qvel[0])
        print('avg time steps:', np.array(sim_time_steps_list).mean())

        rew_history.append(np.mean(rew))
        value_loss_history.append(np.mean(value_loss))
        policy_loss_history.append(np.mean(policy_loss))
        q_loss_history.append(np.mean(q_loss))

        low_rew_history.append(np.mean(low_rew))
        step_history.append(np.array(sim_time_steps_list).mean())
        low_value_loss_history.append(np.mean(low_value_loss))
        low_policy_loss_history.append(np.mean(low_policy_loss))
        low_q_loss_history.append(np.mean(low_q_loss))

        transfer_function_rew_history.append(np.mean(low_rew))
        transfer_function_value_loss_history.append(np.mean(transfer_function_value_loss))
        transfer_function_policy_loss_history.append(np.mean(transfer_function_policy_loss))
        transfer_function_q_loss_history.append(np.mean(transfer_function_q_loss))


        if episode % save_after_episodes == 0 and episode!=0:
            log_all(high_level_controller, path, q_loss_history, policy_loss_history, rew_history, episode, save_network='high_level')
            log_all(step_predictor, path, low_q_loss_history, low_policy_loss_history, low_rew_history, episode,  additional_plot=dict(data=step_history, name='step_plot'), save_network='low_level')
            log_all(transfer_function, path, transfer_function_q_loss_history, transfer_function_policy_loss_history, transfer_function_rew_history, episode, save_network='transfer_function')

        



    return l_vars, labels

    # TODO: plot latent space
        

if __name__ == "__main__":

    '''
    Define inference model trained on encoder
    '''
    inference_path = '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_11_10_26_00_default_true_gmm'
    
    '''
    Define low-level policy for new agent
    '''
    # complex_agent = dict(
    #     environment = WalkerMulti,
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'walker_full_06_07',
    #     epoch = 2600,
    # )
    complex_agent = dict(
        environment = HopperMulti,
        experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
        experiment_name = 'hopper_full_sac0.2_reward1_randomchange',
        epoch = 1400,
    )
    # complex_agent = dict(
    #     environment = HalfCheetahMixtureEnv,
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'new_cheetah_training/half_cheetah_initial_random',
    #     epoch = 700,
    # )
    

    with open(complex_agent['experiments_repo'] + complex_agent['experiment_name'] + '/config.json', 'r') as file:
        env_config = json.load(file)

    env = complex_agent['environment'](env_config)
    env.render_mode = 'rgb_array'

    with open(f'{inference_path}/variant.json', 'r') as file:
        variant = json.load(file)

    m = variant['algo_params']['sac_layer_size']
    simple_env = ENVS[variant['env_name']](**variant['env_params'])         # Just used for initilization purposes

    ### PARAMETERS ###
    obs_dim = int(np.prod(simple_env.observation_space.shape))
    action_dim = int(np.prod(simple_env.action_space.shape))
    net_complex_enc_dec = variant['reconstruction_params']['net_complex_enc_dec']
    latent_dim = variant['algo_params']['latent_size']
    time_steps = variant['algo_params']['time_steps']
    num_classes = variant['reconstruction_params']['num_classes']
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

    memory = Memory(1e+6)
    encoder = get_encoder(inference_path, shared_dim, encoder_input_dim)
    transfer_function = get_complex_agent(env, complex_agent, train=True)

    high_level_controller = SAC(n_states=env.observation_space.shape[0],
                n_actions=env.action_space.shape[0],
                task_dim = variant['algo_params']['latent_size'],
                out_actions = env.task.shape[0],
                hidden_layers_actor = [300,300,300,300],
                hidden_layers_critic = [300,300,300,300],
                memory_size=1e+6,
                batch_size=512,
                gamma=0.9,
                alpha=0.2,
                lr=3e-4,
                action_bounds=[-50,50],
                reward_scale=5)
    output_action_dim = 8

    decoder = None
    step_predictor = SAC(n_states=env.observation_space.shape[0],
                n_actions=1,
                task_dim = 2,   # desired state
                hidden_layers_actor = [64,64,64,64,64],
                hidden_layers_critic = [64,64,64,64,64],
                memory_size=1e+6,
                batch_size=512,
                gamma=0.9,
                alpha=0.2,
                lr=3e-4,
                action_bounds=[-50,50],
                reward_scale=5)

    rollout(env, encoder, decoder, high_level_controller, step_predictor,
                                        transfer_function, variant, obs_dim, action_dim, 
                                        max_path_len, n_tasks=1, inner_loop_steps=10, save_video_path=inference_path)