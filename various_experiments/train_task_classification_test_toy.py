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

from mrl_analysis.plots.plot_settings import *

from agent import SAC
from model import ValueNetwork, QvalueNetwork, PolicyNetwork

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = 'cuda'
ptu.set_gpu_mode(True)

# TODO: einheitliches set to device
simple_env_dt = 0.05
sim_time_steps = 10
max_path_len=50

loss_criterion = nn.CrossEntropyLoss()



# TODO: self.task_dim, task_logits_dim, batch_size are set manually
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


def log_all(agent, path, q1_loss, policy_loss, rew, episode):
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
    def save_plot(loss_history, name:str, path=f'{os.getcwd()}/evaluation/transfer_function/one-sided/'):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        plt.figure()
        # Plotting the loss
        plt.plot(loss_history)
        plt.title('Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(path+name+'.png')
        plt.savefig(path+name+'.pdf')

        plt.close()
        
    # Save networks
    curr_path = path + '/models/policy_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'epoch_{episode}.pth'
    if episode % 500 == 0:
        torch.save(agent.policy_network.cpu(), save_path)
    curr_path = path + '/models/vf1_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'epoch_{episode}.pth'
    if episode % 500 == 0:
        torch.save(agent.q_value_network1.cpu(), save_path)
    curr_path = path + '/models/vf2_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'epoch_{episode}.pth'
    if episode % 500 == 0:
        torch.save(agent.q_value_network2.cpu(), save_path)
    curr_path = path + '/models/value_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'epoch_{episode}.pth'
    if episode % 500 == 0:
        torch.save(agent.value_network.cpu(), save_path)
    agent.q_value_network1.to(DEVICE) 
    agent.q_value_network2.to(DEVICE)
    agent.value_network.to(DEVICE)
    agent.policy_network.to(DEVICE) 

    # Save plots
    path_plots = path + '/plots/'
    save_plot(q1_loss, name='vf_loss', path=path_plots)
    save_plot(rew, name='reward_history', path=path_plots)
    save_plot(policy_loss, name='policy_loss', path=path_plots)

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

def sample_task():
    goal_vel = np.random.choice([0,1,2,3])
    if goal_vel == 0:
        task = np.array([np.random.random()*15 + 1.0,0,0,0,0])
    elif goal_vel == 1:
        task = np.array([np.random.random()*15 - 16,0,0,0,0])
    elif goal_vel == 2:
        task = np.array([0,0,0,np.random.random()*2 + 1, 0])
    else:
        task = np.array([0,0,0,np.random.random()*2 - 3, 0])
    return task

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

def scale_simple_action(simple_action, obs, pos_x=[0.5,25], velocity_x=[0.5, 3.0], pos_y=[np.pi / 5., np.pi / 2.], velocity_y=[2. * np.pi, 4. * np.pi], velocity_z=[1.5, 3.], step='set_position'):
    simple_env_dt = 0.05
    scaled_action = torch.zeros(2)
    scaled_action[0] = obs[0] + simple_action[0]*velocity_x[1]*simple_env_dt
    scaled_action[1] = simple_action[0]*velocity_x[1]

    return scaled_action


def _frames_to_gif(video: cv2.VideoWriter, frames: List[np.ndarray], info, gif_path, transform: Callable = None):
    """ Write collected frames to video file """
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    with imageio.get_writer(gif_path, mode='I', fps=10) as writer:
        for i, frame in enumerate(frames):
            frame = frame.astype(np.uint8)  # Ensure the frame is of type uint8
            frame = np.ascontiguousarray(frame)
            cv2.putText(frame, 'reward: ' + str(info['reward'][i]), (0, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            cv2.putText(frame, 'obs: ' + str(info['obs'][i]), (0, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            cv2.putText(frame, 'simple_action: ' + str(info['simple_action'][i]), (0, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
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
    # output_action_dim = 8
    decoder = ExtendedDecoderMDP(
        action_dim,
        obs_dim,
        reward_dim,
        latent_dim,
        output_action_dim,
        net_complex_enc_dec,
        variant['env_params']['state_reconstruction_clip'],
    ) 

    # decoder.load_state_dict(torch.load(name, map_location='cpu'))

    # for param in decoder.parameters():
    #     param.requires_grad = False
    # for param in decoder.task_decoder.last_fc.parameters():
    #     param.requires_grad = True

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

def step_cheetah(task, obs):

    for i in range(sim_time_steps):
            
            complex_action = transfer_function.get_action(ptu.from_numpy(obs), task, return_dist=False)
            next_obs, r, internal_done, truncated, env_info = env.step(complex_action.detach().cpu().numpy())

            # image = env.render()
            # frames.append(image)
            # image_info['reward'].append(r)
            # image_info['obs'].append(task)
            # image_info['base_task'].append(env.task)
            # if internal_done:
            #     break

            obs = next_obs
        
    return r, next_obs

def normalize_data(stats_dict, o, a, r, next_o):
        o = torch.Tensor((o - stats_dict['observations']['mean']) / (stats_dict['observations']['std'] + 1e-9))
        a = torch.Tensor((a - stats_dict['actions']['mean']) / (stats_dict['actions']['std'] + 1e-9))
        r = torch.Tensor((r - stats_dict['rewards']['mean']) / (stats_dict['rewards']['std'] + 1e-9))
        next_o = torch.Tensor((next_o - stats_dict['next_observations']['mean']) / (stats_dict['next_observations']['std'] + 1e-9))

        return o, a, r, next_o


def rollout(env, encoder, decoder, optimizer, simple_agent, transfer_function, memory, 
            variant, obs_dim, actions_dim, max_path_len, 
            n_tasks, inner_loop_steps, save_video_path, experiment_name, tasks=None):
    range_dict = OrderedDict(pos_x = [0.5, 25],
                             velocity_z = [1.5, 3.],
                             pos_y = [np.pi / 6., np.pi / 2.],
                             velocity_x = [0.5, 3.0],
                             velocity_y = [2. * np.pi, 4. * np.pi],
                             )
    
    save_after_episodes = 5
    value_loss_history, q_loss_history, policy_loss_history, rew_history = [], [], [], []
    path = save_video_path
    
    # with open(f'{save_video_path}/weights/stats_dict.json', 'r') as file:
    #     stats_dict = json.load(file)

    loss_history = []


    tasks_pos, tasks_vel = [], []
    x_pos_plot, x_vel_plot = [],[]
    episodes = 100
    if tasks:
        episodes=len(tasks)
    for episode in range(episodes):

        
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
            
        print(episode)

        path_length = 0
        obs = env.reset()[0]
        _simple_obs  = simple_env.reset()[0]
        x_pos_curr.append(env.sim.data.qpos[0])
        x_vel_curr.append(env.sim.data.qvel[0])
        # simple_env.reset_model()
        contexts = torch.zeros((n_tasks, variant['algo_params']['time_steps'], obs_dim + 1 + obs_dim), device=DEVICE)
        l_vars = []
        labels = []

        done = 0
        episode_reward = 0
        loss = []
        value_loss, q_loss, policy_loss = [], [], []
        task = env.sample_task(test=True, task=tasks[episode])
        # env.update_task(task)

        _loss  = []
        if env.base_task == env.config.get('tasks',{}).get('goal_front'):
            simple_env.base_task = 2
        elif env.base_task == env.config.get('tasks',{}).get('goal_back'):
            simple_env.base_task = 3
        elif env.base_task == env.config.get('tasks',{}).get('forward_vel'):
            simple_env.base_task = 0
        elif env.base_task == env.config.get('tasks',{}).get('backward_vel'):
            simple_env.base_task = 1
        simple_env.task_specification = env.task[env.base_task]
        for path_length in range(max_path_len):


            _simple_obs_before = _simple_obs

            # get encodings
            encoder_input = contexts.detach().clone()
            encoder_input = encoder_input.view(encoder_input.shape[0], -1).to(DEVICE)
            mu, log_var = encoder(encoder_input)     # Is this correct??

            # Save latent vars
            policy_input = torch.cat([ptu.from_numpy(_simple_obs), mu.squeeze()], dim=-1)
            simple_action = simple_agent.get_torch_actions(policy_input, deterministic=True)
            
            # _,_, logits = decoder(ptu.from_numpy(simple_obs_before), simple_action, 0, mu.squeeze())
            _simple_obs,r,_,_ = simple_env.step(simple_action.detach().cpu().numpy())
            image = simple_env.render()
            imageio.imwrite('{os.getcwd()}/toy.pdf', image)
            # simple_obs = torch.zeros_like(torch.tensor(task)).to(DEVICE)

            x_pos_curr.append(simple_env.sim.data.qpos[0])
            x_vel_curr.append(simple_env.sim.data.qvel[0])

            episode_reward += r
            task_idx = env.base_task

            task_idx = torch.tensor([task_idx]).to("cpu")
            # memory.add(task_idx, ptu.from_numpy(simple_obs_before), simple_action, mu.squeeze())
            simple_obs_after = _simple_obs
            
            data = torch.cat([ptu.from_numpy(_simple_obs_before), torch.unsqueeze(torch.tensor(r, device=DEVICE), dim=0), ptu.from_numpy(simple_obs_after)], dim=0).unsqueeze(dim=0)
            context = torch.cat([contexts.squeeze(), data], dim=0)
            contexts = context[-time_steps:, :]
            contexts = contexts.unsqueeze(0).to(torch.float32)

            # if len(memory) < memory.batch_size:
            #     continue
            # else: 
            #     batch = memory.sample(memory.batch_size)
            #     tasks_batch, simple_obs_batch, simple_action_batch, mu_batch = memory.unpack(batch)
            #     _,_, logits_batch = decoder(simple_obs_batch.detach(), simple_action_batch.detach(), 0, mu_batch.squeeze().detach())
            #     loss = loss_criterion(logits_batch.squeeze(), tasks_batch.squeeze().detach())
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()

            #     _loss.append(loss)
            _loss = []


        if env.base_task in [env.config.get('tasks',{}).get('goal_front'), env.config.get('tasks',{}).get('goal_back')]:
            tasks_pos.append(simple_env.task_specification)
            x_pos_plot.append(x_pos_curr)
        elif env.base_task in [env.config.get('tasks',{}).get('forward_vel'), env.config.get('tasks',{}).get('backward_vel')]:
            x_vel_plot.append(x_vel_curr)
            tasks_vel.append(simple_env.task_specification)
        if len(x_pos_plot) > len(tasks):
            x_pos_plot.pop(0)
            tasks_pos.pop(0)
        if len(x_vel_plot) > len(tasks):
            x_vel_plot.pop(0)
            tasks_vel.pop(0)


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
        dir = Path(os.path.join(save_video_path, 'toy_test', 'trajectories_plots'))
        filename = os.path.join(dir,f"epoch_{episode}.png")
        filename = os.path.join(dir,f"epoch_{episode}.pdf")
        dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(filename, dpi=300)  # Save the figure with 300 dpi resolution
        plt.close()

        if len(_loss)>0:
            loss_history.append(torch.stack(_loss).mean().detach().cpu().numpy())


        if episode % save_after_episodes == 0 and episode!=0:
            file = os.path.join(save_video_path, 'weights/retrained_decoder.pth')
            torch.save(decoder.state_dict(), file)

        # if video:
        #     save_plot(np.array(loss_history), name='task_loss', path=save_video_path)
        #     size = frames[0].shape

        #     # Save to corresponding repo
        #     fps=20
        #     save_as = f'{save_video_path}/videos/transfer_{episode}.mp4'
        #     # video = cv2.VideoWriter(save_as, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
        #     # Write frames to video
        #     _frames_to_gif(video, frames, image_info, save_as)
        #     # video.release()

        

if __name__ == "__main__":
    # from experiments_configs.half_cheetah_multi_env import config as env_config

    inference_path = '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_15_34_37_default_true_gmm'
    paths = [
        '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_12_22_59_default_true_gmm',
        '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_15_34_37_default_true_gmm',
        '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_02_15_23_09_default_true_gmm',
        '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_11_10_26_00_default_true_gmm',
        '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_11_11_49_42_default_true_gmm',
        '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_12_09_17_42_default_true_gmm',
        '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_12_20_31_24_default_true_gmm',
        '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_14_20_55_42_default_true_gmm',
        '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_12_23_15_default_true_gmm',
        '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_15_34_37_default_true_gmm',
        # {'name': 'inference_path_3', 'path': '/path/to/inference_path_3'},
    ]
    tasks = [
            {'base_task':'goal_back', 'specification':1},
            {'base_task':'goal_back', 'specification':0.75},
            {'base_task':'goal_back', 'specification':0.5},
            {'base_task':'goal_front', 'specification':0.5},
            {'base_task':'goal_front', 'specification':0.75},
            {'base_task':'goal_front', 'specification':1},
            {'base_task':'backward_vel', 'specification':1},
            {'base_task':'backward_vel', 'specification':0.5},
            {'base_task':'backward_vel', 'specification':0.25},
            # {'base_task':'backward_vel', 'specification':1.0},
            # {'base_task':'forward_vel', 'specification':1.0},
            {'base_task':'forward_vel', 'specification':0.25},
            {'base_task':'forward_vel', 'specification':0.5},
            {'base_task':'forward_vel', 'specification':1},
                 ]
    for inference_path in paths:

            complex_agent_config = dict(
                experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
                experiment_name = 'new_cheetah_training/half_cheetah_back_pos_vel',
                epoch = 100,
            )

            with open(complex_agent_config['experiments_repo'] + complex_agent_config['experiment_name'] + '/config.json', 'r') as file:
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
            env.render_mode = 'rgb_array'

            with open(f'{inference_path}/variant.json', 'r') as file:
                variant = json.load(file)

            # ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])

            m = variant['algo_params']['sac_layer_size']
            simple_env = ENVS[variant['env_name']](**variant['env_params'])         # Just used for initilization purposes
            simple_env.render_mode = 'rgb_array'

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

            memory = Memory(1e+6)
            encoder = get_encoder(inference_path, shared_dim, encoder_input_dim)
            simple_agent = get_simple_agent(inference_path, obs_dim, policy_latent_dim, action_dim, m)
            transfer_function = get_complex_agent(env, complex_agent_config)
            output_action_dim = env.task.shape[0]
            decoder = get_decoder(inference_path, action_dim, obs_dim, reward_dim, latent_dim, output_action_dim, net_complex_enc_dec, variant)
            optimizer = optim.Adam(decoder.parameters(), lr=3e-4)

            ### ROLLOUT ###
            # for i, task in enumerate(tasks):
            #     simple_env.reset_model()
            #     if i == 0:
            #         res = rollout(task, env, encoder, decoder, simple_agent, high_level_controller,
            #                                     transfer_function, variant, obs_dim, action_dim, 
            #                                     max_path_len, n_tasks=1, inner_loop_steps=10, save_video_path=inference_path)
            #         latent_vars = res[0]
            #         labels = res[1]
            #     else:
            #         res = rollout(task, env, encoder, decoder, simple_agent, high_level_controller, 
            #                                     transfer_function, variant, obs_dim, action_dim, 
            #                                     max_path_len, n_tasks=1, inner_loop_steps=10, save_video_path=inference_path)
            #         latent_vars = np.concatenate((latent_vars, res[0]), axis = 0)
            #         labels = np.concatenate((labels, res[1]), axis=0)
            rollout(env, encoder, decoder, optimizer, simple_agent,
                                                transfer_function, memory, variant, obs_dim, action_dim, 
                                                max_path_len, n_tasks=1, inner_loop_steps=10, save_video_path=inference_path, experiment_name=complex_agent_config['experiment_name'], tasks=tasks)
            ### Save metadata, tensors, videos
            # create_tsne(np.array(latent_vars), np.array(labels), inference_path)