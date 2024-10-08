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
from vis_utils.logging import log_all, _frames_to_gif

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = 'cuda'
ptu.set_gpu_mode(True)

# TODO: einheitliches set to device
simple_env_dt = 0.05
sim_time_steps = 10
max_path_len = 100
num_trajectories = 50
plot_every = 5
loss_criterion = nn.CrossEntropyLoss()

class Memory():

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = []
        self.Transition = namedtuple('Transition',
                        ('task', 'simple_obs', 'simple_action', 'mu'))
        self.batch_size = 256
        self.task_dim = 1
        self.latent_dim = 4
        self.simple_obs_dim = 2
        self.simple_action_dim = 1

    def add(self, *transition):
        self.memory.append(self.Transition(*transition))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        assert len(self.memory) <= self.memory_size

    def sample(self, size):
        return random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)
    
    def unpack(self, batch):
        batch = self.Transition(*zip(*batch))
        
        tasks = torch.cat(batch.task).view(self.batch_size, self.task_dim).to(DEVICE)
        simple_obs = torch.cat(batch.simple_obs).view(self.batch_size, self.simple_obs_dim).to(DEVICE)
        simple_action = torch.cat(batch.simple_action).view(self.batch_size, self.simple_action_dim).to(DEVICE)
        mu = torch.cat(batch.mu).view(self.batch_size, self.latent_dim).to(DEVICE)

        return tasks, simple_obs, simple_action, mu

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
    encoder.to(DEVICE)
    return encoder

def get_simple_agent(path, obs_dim, policy_latent_dim, action_dim, m):
    path = os.path.join(path, 'weights')
    for filename in os.listdir(path):
        if filename.startswith('policy'):
            name = os.path.join(path, filename)
    
    policy = TanhGaussianPolicy(
        obs_dim=(obs_dim + policy_latent_dim),
        action_dim=action_dim,
        latent_dim=policy_latent_dim,
        hidden_sizes=[m,m,m],
    )
    policy.load_state_dict(torch.load(name, map_location='cpu'))
    policy.to(DEVICE)
    return policy

def get_complex_agent(env, complex_agent_config):
    pretrained = complex_agent_config['experiments_repo']+complex_agent_config['experiment_name']+f"/models/policy_model/epoch_{complex_agent_config['epoch']}.pth"
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

def cheetah_to_simple_env_obs(obs):
    simple_observations = np.zeros(obs_dim)
    simple_observations[...,0] = obs[...,-3]
    # simple_observations[...,1:3] = obs[...,1:3]
    # simple_observations[...,3:] = obs[...,7:10]
    simple_observations[...,1] = obs[...,8]
    return simple_observations

def general_obs_map(env):
    simple_observations = np.zeros(obs_dim)
    simple_observations[...,0] = env.sim.data.qpos[0]    
    simple_observations[...,1] = env.sim.data.qvel[0]    
    return simple_observations

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

    for param in decoder.parameters():
        param.requires_grad = False
    for param in decoder.task_decoder.last_fc.parameters():
        param.requires_grad = True

    decoder.to(DEVICE)
    return decoder

def create_tsne(latent_variables, task_labels, path):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    save_as = os.path.join(path , 'tsne_test.png')
    save_as = os.path.join(path , 'tsne_test.pdf')
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(latent_variables)

    # Plot
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(task_labels)
    for label in unique_labels:
        idx = task_labels == label
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=label, alpha=0.7)
    plt.legend()
    plt.title('t-SNE Visualization of Latent Variables')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(save_as)
    plt.close()

def save_plot(loss_history, name:str, path=f'{os.getcwd()}/plots'):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        plt.figure()
        # Plotting the loss
        plt.plot(loss_history)
        plt.title('Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(path,name+'.png'))
        plt.savefig(os.path.join(path,name+'.pdf'))

        plt.close()


def rollout(env, encoder, decoder, optimizer, simple_agent, step_predictor, transfer_function, memory, 
            variant, obs_dim, actions_dim, max_path_len, 
            n_tasks, inner_loop_steps, save_video_path, experiment_name,
            current_inference_path_name, tasks=None, beta=0.1):
    range_dict = OrderedDict(pos_x = [0.5, 25],
                             velocity_z = [1.5, 3.],
                             pos_y = [np.pi / 6., np.pi / 2.],
                             velocity_x = [0.5, 3.0],
                             velocity_y = [2. * np.pi, 4. * np.pi],
                             )
    
    save_after_episodes = 5
    value_loss_history, q_loss_history, policy_loss_history, rew_history = [], [], [], []
    path = save_video_path
    
 
    loss_history = []

    pos_reward, vel_reward = 0, 0
    tasks_pos, tasks_vel = [], []
    x_pos_plot, x_vel_plot = [],[]
    if tasks:
        num_trajectory = len(tasks)
        save_after_episodes = 1
    else: 
        num_trajectory= num_trajectories

    for episode in range(num_trajectory):

        
        video = False
        if episode % 1 == 0:
            frames = []
            image_info = dict(reward = [],
            obs = [],
            base_task = [],
            complex_action = [],
            simple_action = [])
            video = True
        

        x_pos_curr, x_vel_curr = [],[]
            
        print(f"Inference Path: {current_inference_path_name}, Episode: {episode}")

        path_length = 0
        obs = env.reset()[0]
        x_pos_curr.append(env.sim.data.qpos[0])
        x_vel_curr.append(env.sim.data.qvel[0])
        simple_env.reset_model()
        contexts = torch.zeros((n_tasks, variant['algo_params']['time_steps'], obs_dim + 1 + obs_dim), device=DEVICE)
        l_vars = []
        labels = []

        done = 0
        episode_reward = 0
        loss = []
        value_loss, q_loss, policy_loss = [], [], []
        if tasks:
            task = env.sample_task(task=tasks[episode], test=True)
        else:
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

        _loss  = []

        sim_time_steps_list = []
        for path_length in range(max_path_len):

            # get encodings
            simple_env.sim.data.set_joint_qpos('boxslideX', env.sim.data.qpos[0])
            simple_env.sim.data.set_joint_qvel('boxslideX', env.sim.data.qvel[0])
            encoder_input = contexts.detach().clone()
            encoder_input = encoder_input.view(encoder_input.shape[0], -1).to(DEVICE)
            mu, log_var = encoder(encoder_input)

            obs_before_sim = env._get_obs()
            simple_obs_before = general_obs_map(env)

            # Save latent vars
            policy_input = torch.cat([ptu.from_numpy(simple_obs_before), mu.squeeze()], dim=-1)
            simple_action = simple_agent.get_torch_actions(policy_input, deterministic=True)
            
            _,_, logits = decoder(ptu.from_numpy(simple_obs_before), simple_action, 0, mu.squeeze())
            task_prediction = torch.argmax(torch.nn.functional.softmax(logits), dim=0)
            # task_prediction = 0

            _simple_obs,_,_,_ = simple_env.step(simple_action.detach().cpu().numpy())
            simple_obs = torch.zeros_like(torch.tensor(task)).to(DEVICE)

            if task_prediction in [env.config.get('tasks',{}).get('goal_front'), env.config.get('tasks',{}).get('goal_back')]:
                if simple_action.item()>0:
                    simple_obs[env.config.get('tasks',{}).get('goal_front')] = _simple_obs[0].item()
                else:
                    simple_obs[env.config.get('tasks',{}).get('goal_back')] = _simple_obs[0].item()
            else:
                if simple_action.item()>0:
                    simple_obs[env.config.get('tasks',{}).get('forward_vel')] = np.clip(_simple_obs[1].item(), -3,3)
                else:
                    simple_obs[env.config.get('tasks',{}).get('backward_vel')] = np.clip(_simple_obs[1].item(), -3,3)


            base_task_pred = torch.argmax(torch.abs(simple_obs))
            action = torch.zeros_like(simple_obs).to(DEVICE)
            action[base_task_pred] = simple_obs[base_task_pred]
            desired_state = action
            action_normalize = simple_obs_before[0].item() - simple_obs[0].item()
            max_steps = 20
            sim_time_steps = int(torch.clamp(step_predictor.choose_action(obs, desired_state.detach().cpu().numpy(), torch=True).squeeze()*max_steps,1,max_steps))
            for i in range(sim_time_steps):
                complex_action = transfer_function.get_action(ptu.from_numpy(obs), simple_obs, return_dist=False)
                next_obs, r, done, truncated, env_info = env.step(complex_action.detach().cpu().numpy(), healthy_scale = 0)
                obs = next_obs
                if video:
                    image = env.render()
                    frames.append(image)
                    image_info['reward'].append(r)
                    image_info['obs'].append(cheetah_to_simple_env_map(obs_before_sim, obs)[0])
                    image_info['base_task'].append(env.task)
                    image_info['complex_action'].append(complex_action)
                    image_info['simple_action'].append(simple_obs)


                episode_reward += r
            simple_env.sim.data.set_joint_qpos('boxslideX', env.sim.data.qpos[0])
            simple_env.sim.data.set_joint_qvel('boxslideX', env.sim.data.qvel[0])
            x_pos_curr.append(env.sim.data.qpos[0])
            x_vel_curr.append(env.sim.data.qvel[0])
            task_idx = env.base_task
            task_idx = torch.tensor([task_idx]).to("cpu")
            # memory.add(task_idx, ptu.from_numpy(simple_obs_before), simple_action, mu.squeeze())

            sim_time_steps_list.append(sim_time_steps)

            if base_task_pred in [env.config.get('tasks',{}).get('goal_front')]:
                low_level_r = - np.abs(action[env.config['tasks']['goal_front']].detach().cpu().numpy()-env.sim.data.qpos[0])/np.abs(action_normalize)- beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('goal_back')]:
                low_level_r = - np.abs(action[env.config['tasks']['goal_back']].detach().cpu().numpy()-env.sim.data.qpos[0] )/np.abs(action_normalize)- beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('forward_vel')]:
                low_level_r = - np.abs(action[env.config['tasks']['forward_vel']].detach().cpu().numpy()-env.sim.data.qvel[0])/np.abs(action[env.config['tasks']['forward_vel']].item())- beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('backward_vel')]:
                low_level_r = - np.abs(action[env.config['tasks']['backward_vel']].detach().cpu().numpy()-env.sim.data.qvel[0])/np.abs(action[env.config['tasks']['backward_vel']].item())- beta * sim_time_steps
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
            desired_state = desired_state.to(torch.float32)
            step_predictor.store(obs_before_sim, low_level_r, done, np.array([sim_time_steps]), next_obs, desired_state)
            low_level_losses = step_predictor.train(episode, False)

            simple_obs_after = general_obs_map(env)
            
            data = torch.cat([ptu.from_numpy(simple_obs_before), torch.unsqueeze(torch.tensor(r, device=DEVICE), dim=0), ptu.from_numpy(simple_obs_after)], dim=0).unsqueeze(dim=0)
            context = torch.cat([contexts.squeeze(), data], dim=0)
            contexts = context[-time_steps:, :]
            contexts = contexts.unsqueeze(0).to(torch.float32)

            if len(memory) < memory.batch_size:
                continue
            else: 
                batch = memory.sample(memory.batch_size)
                tasks_batch, simple_obs_batch, simple_action_batch, mu_batch = memory.unpack(batch)
                _,_, logits_batch = decoder(simple_obs_batch.detach(), simple_action_batch.detach(), 0, mu_batch.squeeze().detach())
                loss = loss_criterion(logits_batch.squeeze(), tasks_batch.squeeze().detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _loss.append(loss)


        # Collect rewards for box plots
        if env.base_task in [env.config.get('tasks',{}).get('goal_front'), env.config.get('tasks',{}).get('goal_back')]:
            # Position task
            rewards_data[current_inference_path_name]['position'].append(episode_reward)
        elif env.base_task in [env.config.get('tasks',{}).get('forward_vel'), env.config.get('tasks',{}).get('backward_vel')]:
            # Velocity task
            rewards_data[current_inference_path_name]['velocity'].append(episode_reward)

        if len(_loss)>0:
            loss_history.append(torch.stack(_loss).mean().detach().cpu().numpy())


        if episode % save_after_episodes == 0 and episode!=0:
            file = os.path.join(save_video_path, 'weights/retrained_decoder.pth')
            torch.save(decoder.state_dict(), file)
        # video = False
        if video:
            # save_plot(np.array(loss_history), name='task_loss', path=f'{os.getcwd()}/delete_videos')
            size = frames[0].shape

            # Save to corresponding repo
            fps=20
            save_as = f'{save_video_path}/videos/transfer_{episode}.mp4'
            # video = cv2.VideoWriter(save_as, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
            # Write frames to video
            _frames_to_gif(video, frames, image_info, save_as)
            # video.release()

        if env.base_task in [env.config.get('tasks',{}).get('goal_front'), env.config.get('tasks',{}).get('goal_back')]:
            tasks_pos.append(task[env.base_task])
            x_pos_plot.append(x_pos_curr)
            pos_reward += np.clip(episode_reward, -2*path_length*sim_time_steps, 0)
        elif env.base_task in [env.config.get('tasks',{}).get('forward_vel'), env.config.get('tasks',{}).get('backward_vel')]:
            x_vel_plot.append(x_vel_curr)
            tasks_vel.append(task[env.base_task])
            vel_reward += episode_reward
        if tasks is None:
            if len(x_pos_plot) > 5:
                x_pos_plot.pop(0)
                tasks_pos.pop(0)
            if len(x_vel_plot) > 5:
                x_vel_plot.pop(0)
                tasks_vel.pop(0)

        if episode%plot_every == 0 or tasks:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']
            window_size = 10

            def moving_average(data, window_size):
                # """Compute the moving average using a sliding window."""
                # window = np.ones(int(window_size))/float(window_size)
                # return np.convolve(data, window, 'same')
                from scipy.ndimage.filters import gaussian_filter1d
                return gaussian_filter1d(data, sigma=2)

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
            # ax1.set_title(f'Avg Reward: {pos_reward/(episode+1)}')

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
            plt.subplots_adjust(hspace=0.4)

            # Save the figure to a file
            dir = Path(os.path.join('{os.getcwd()}/trajectories', experiment_name, current_inference_path_name, 'beta0.1_old'))
            filename = os.path.join(dir,f"epoch_{episode}.png")
            filename = os.path.join(dir,f"epoch_{episode}.pdf")

            if tasks:
                filename = os.path.join(dir,f"final.png")
                filename = os.path.join(dir,f"final_{max_steps}.pdf")
            dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(filename, dpi=300)  # Save the figure with 300 dpi resolution
            plt.close()
        

if __name__ == "__main__":
    # from experiments_configs.half_cheetah_multi_env import config as env_config

    '''
    List of inference paths to test. The inference paths contain the inference models trained on the toy with different randomization values
    '''
    inference_paths = [
        # {'name': 'var_0.1keep', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_15_34_37_default_true_gmm'},
        # {'name': 'var_0.1keepno', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_12_22_59_default_true_gmm'},
        # {'name': 'var_0.02', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_21_11_06_00_default_true_gmm'},
        # {'name': 'var_0.05', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_23_12_40_04_default_true_gmm'},
        {'name': 'var_0.1', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_15_34_37_default_true_gmm'},
        {'name': 'no_var', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_02_15_23_09_default_true_gmm'},
        # {'name': 'no random step 10', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_11_11_49_42_default_true_gmm'},
        {'name': 'var_0.2', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_12_09_17_42_default_true_gmm'},
        # {'name': 'var_0.01', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_12_20_31_24_default_true_gmm'},
        # {'name': 'var_0.1, step 10', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_14_20_55_42_default_true_gmm'},
        # {'name': 'var_0.1.2, step 10', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_12_23_15_default_true_gmm'},
        # {'name': 'var_0.1.3, step 10', 'path': '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_15_34_37_default_true_gmm'},
        # {'name': 'inference_path_3', 'path': '/path/to/inference_path_3'},
    ]

    '''
    Define the low-level policy and agent to test
    '''
    complex_agent_config = dict(
        environment = HalfCheetahMixtureEnv,
        experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
        experiment_name = 'new_cheetah_training/half_cheetah_initial_random',
        epoch = 700,
    )
    # complex_agent_config = dict(
    #     environment = HopperMulti,
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'hopper_full_sac0.2_reward1_randomchange',
    #     epoch = 1400,
    # )
    # complex_agent_config = dict(
    #     environment = HopperMulti,
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'hopper_12_07',
    #     epoch = 2500,
    # )
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
                device=DEVICE
                # pretrained=dict(path='/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_25_12_38_14_default_true_gmm/cheetah_26_08', epoch='low_level')
                )

        memory = Memory(1e+6)
        encoder = get_encoder(inference_path, shared_dim, encoder_input_dim)
        simple_agent = get_simple_agent(inference_path, obs_dim, policy_latent_dim, action_dim, m)
        transfer_function = get_complex_agent(env, complex_agent_config)
        output_action_dim = env.task.shape[0]
        decoder = get_decoder(inference_path, action_dim, obs_dim, reward_dim, latent_dim, output_action_dim, net_complex_enc_dec, variant)
        optimizer = optim.Adam(decoder.parameters(), lr=3e-4)

        ### ROLLOUT ###
        rollout(env, encoder, decoder, optimizer, simple_agent,  step_predictor,
                                        transfer_function, memory, variant, obs_dim, action_dim, 
                                        max_path_len, n_tasks=1, inner_loop_steps=10, save_video_path=inference_path, experiment_name=complex_agent_config['experiment_name'],
                                        current_inference_path_name=current_inference_path_name)
       
        '''
        After the striding predictor is trained, plot the results with symmetric goals
        '''
        tasks = [
            {'base_task':'goal_back', 'specification':0.9},
            {'base_task':'goal_back', 'specification':0.5},
            {'base_task':'goal_back', 'specification':0.3},
            {'base_task':'goal_front', 'specification':0.3},
            {'base_task':'goal_front', 'specification':0.5},
            {'base_task':'goal_front', 'specification':0.9},
            {'base_task':'backward_vel', 'specification':0.9},
            {'base_task':'backward_vel', 'specification':0.5},
            {'base_task':'backward_vel', 'specification':0.1},
            # {'base_task':'backward_vel', 'specification':1.0},
            # {'base_task':'forward_vel', 'specification':1.0},
            {'base_task':'forward_vel', 'specification':0.1},
            {'base_task':'forward_vel', 'specification':0.5},
            {'base_task':'forward_vel', 'specification':0.9},
                 ]
        rollout(env, encoder, decoder, optimizer, simple_agent, step_predictor,
                                        transfer_function, memory, variant, obs_dim, action_dim, 
                                        max_path_len, n_tasks=1, inner_loop_steps=10, save_video_path=inference_path, experiment_name=complex_agent_config['experiment_name'],
                                        current_inference_path_name=current_inference_path_name, tasks=tasks)


    '''
    Create the box plot
    '''
    import matplotlib.pyplot as plt

    # Assuming inference_paths and rewards_data are already defined

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
        pos += 3  # Adding space between groups

    # Create the box plot and retrieve the dictionary of artists
    plt.figure(figsize=(12, 6))
    box = plt.boxplot(boxplot_data, positions=positions, widths=0.6, showfliers=False, patch_artist=True)

    # Customize median colors
    medians = box['medians']
    for i, median in enumerate(medians):
        if i % 2 == 0:  # Even index: Position
            median.set_color('blue')    # Set color for position medians
            median.set_linewidth(2)     # Optional: set line width
        else:           # Odd index: Velocity
            median.set_color('green')     # Set color for velocity medians
            median.set_linewidth(2)     # Optional: set line width

    # Optional: Customize box colors for better visualization
    boxes = box['boxes']
    for i, box_patch in enumerate(boxes):
        if i % 2 == 0:  # Even index: Position
            box_patch.set_facecolor('#ADD8E6')  # Light blue
        else:           # Odd index: Velocity
            box_patch.set_facecolor('#90EE90')  # Light pink

    # Set x-axis labels
    plt.xticks(positions, x_labels, rotation=45, ha='right')

    # Set y-axis label
    plt.ylabel('Rewards')

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.5)

    # Optional: Add a legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ADD8E6', edgecolor='blue', label='Position'),
                    Patch(facecolor='#90EE90', edgecolor='green', label='Velocity')]
    plt.legend(handles=legend_elements, loc='lower right')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{os.getcwd()}/rewards_boxplot.png', dpi=300)
    plt.savefig(f'{os.getcwd()}/rewards_boxplot.pdf', dpi=300)

    # Show the plot
    plt.show()