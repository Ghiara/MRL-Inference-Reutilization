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
from replay_memory import Memory
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from agent import SAC
from model import ValueNetwork, QvalueNetwork, PolicyNetwork

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
ptu.set_gpu_mode(False)

experiment_name = 'train_step_predictor_walker'
# TODO: einheitliches set to device
simple_env_dt = 0.05
sim_time_steps = 20
max_path_len=100
save_after_episodes = 5
plot_every = 1
batch_size = 50
policy_update_steps = 2048

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
    def save_plot(loss_history, name:str, path='/home/ubuntu/juan/Meta-RL/evaluation/transfer_function/one-sided/'):
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
        
        # TODO: save both vf losses (maybe with arg)
        os.makedirs(path, exist_ok=True)
        loss_history = remove_outliers_iqr(loss_history)
        smoothed_loss = moving_average(loss_history)

        plt.figure()
        # Plotting the loss
        plt.plot(smoothed_loss, color='blue')
        plt.plot(loss_history, color='blue', alpha=0.3)
        plt.title('Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(path,name+'.png'))

        plt.close()
    
    path = os.path.join(path, experiment_name)
    # Save networks
    curr_path = path + '/models/policy_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'{save_network}.pth'
    if episode % 500 == 0:
        torch.save(agent.policy_network.cpu(), save_path)
    curr_path = path + '/models/vf1_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'{save_network}.pth'
    if episode % 500 == 0:
        torch.save(agent.q_value_network1.cpu(), save_path)
    curr_path = path + '/models/vf2_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'{save_network}.pth'
    if episode % 500 == 0:
        torch.save(agent.q_value_network2.cpu(), save_path)
    curr_path = path + '/models/value_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'{save_network}.pth'
    if episode % 500 == 0:
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


def _frames_to_gif(frames: List[np.ndarray], info, gif_path, transform: Callable = None):
    """ Write collected frames to video file """
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    print(gif_path)
    with imageio.get_writer(gif_path, mode='I', fps=10) as writer:
        for i, frame in enumerate(frames):
            frame = frame.astype(np.uint8)  # Ensure the frame is of type uint8
            frame = np.ascontiguousarray(frame)
            # cv2.putText(frame, 'reward: ' + str(info['reward'][i]), (0, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            # cv2.putText(frame, 'obs: ' + str(info['obs'][i]), (0, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            # cv2.putText(frame, 'action: ' + str(info['action'][i]), (0, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            # cv2.putText(frame, 'task: ' + str(info['base_task'][i]), (0, 75), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
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

def rollout(env, transfer_function, 
            variant, obs_dim, max_path_len, 
            n_tasks,save_video_path, batch_size=batch_size):
    range_dict = OrderedDict(pos_x = [0.5, 25],
                             velocity_z = [1.5, 3.],
                             pos_y = [np.pi / 6., np.pi / 2.],
                             velocity_x = [0.5, 3.0],
                             velocity_y = [2. * np.pi, 4. * np.pi],
                             )
    
    value_loss_history, q_loss_history, policy_loss_history, rew_history = [], [], [], []
    low_value_loss_history, low_q_loss_history, low_policy_loss_history, low_rew_history, step_history = [], [], [], [], []
    path = save_video_path
    
    with open(f'{save_video_path}/weights/stats_dict.json', 'r') as file:
        stats_dict = json.load(file)

    
    x_pos_plot = []
    x_vel_plot = []
    tasks_vel = []
    tasks_pos = []
    l_vars = []
    labels = []

    value_loss, q_loss, policy_loss = [], [], []
    low_value_loss, low_q_loss, low_policy_loss = [], [], []
    print('Collecting samples...')
    plot = True
    frames = []
    image_info = dict(reward = [],
    obs = [],
    base_task = [],
    action = [])

    
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
    task = np.zeros_like(task)


    base_task = env.config.get('tasks',{}).get('backward_vel')
    # base_task = env.config.get('tasks',{}).get('goal_front')
    # base_task = env.config.get('tasks',{}).get('jump')
    # base_task = env.config.get('tasks',{}).get('stand_back')
    task[base_task] = -2.3



    env.update_base_task(base_task)
    env.update_task(task)

    sim_time_steps_list = []

    while not done and path_length < max_path_len:
        path_length += 1

        # get encodings
        encoder_input = contexts.detach().clone()
        encoder_input = encoder_input.view(encoder_input.shape[0], -1).to(DEVICE)

        action = torch.from_numpy(task).to(DEVICE)
        complex_action = transfer_function.get_action(ptu.from_numpy(obs), action, return_dist=False)
        next_obs, step_r, done, truncated, env_info = env.step(complex_action.detach().cpu().numpy())
        simple_obs = [env.sim.data.qpos[0], env.sim.data.qvel[0]]
        
        obs = next_obs
        if plot:
            image = env.render()
            frames.append(image)
            # image_info['reward'].append(r)
            # image_info['obs'].append(cheetah_to_simple_env_map(obs_before_sim, obs)[0])
            image_info['base_task'].append(env.task)
            image_info['action'].append(action)
        r = step_r
        print(env.sim.data.qpos[0], env.sim.data.qvel[0])
        
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

    if plot:
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
            color = colors[i]
            # x_pos = moving_average(x_pos, window_size)
            # Plot position on the first (left) axis
            ax1.plot(np.arange(len(x_pos)), np.array(x_pos), label='Position', color=color)
            # if tasks[i][0]!=0:
            ax1.plot(np.arange(len(x_pos)), np.ones(len(x_pos))*tasks_pos[i], linestyle='--', color=color)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (m)')
        ax1.tick_params(axis='y')

        # Create a second axis sharing the same x-axis
        for i, x_vel in enumerate(x_vel_plot):
            color = colors[i]
            # x_vel = moving_average(x_vel, window_size)
            ax2.plot(np.arange(len(x_vel)), np.array(x_vel), label='Velocity', color=color)
            # if tasks[i][3]!=0:
            ax2.plot(np.arange(len(x_vel)), np.ones(len(x_vel))*tasks_vel[i], linestyle='--', color=color)
        if x_vel_plot:
            ax2.tick_params(axis='y')
            ax2.set_ylabel('Velocity (m/s)')

        # Save the figure to a file
        dir = Path(os.path.join(save_video_path, experiment_name, 'trajectories_plots'))
        filename = os.path.join(dir,f"test_{task}.png")
        dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(filename, dpi=300)  # Save the figure with 300 dpi resolution
        plt.close()

        # Save to corresponding repo
        # fps=10
        save_as = f'{save_video_path}/videos/{experiment_name}/test_{task}.mp4'
        print('Save video under', save_as)
        # video = cv2.VideoWriter(save_as, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
        # Write frames to video
        _frames_to_gif(frames, image_info, save_as)
        # video.release()


    low_rew.append(low_episode_reward)

    rew.append(episode_reward)
    print('Training policies...')

    print('Task:', task[env.base_task], 'Base task:', env.base_task)
    print('END obs:', env.sim.data.qpos[0], env.sim.data.qvel[0])
    # print('avg time steps:', np.array(sim_time_steps_list).mean())


        # if episode % save_after_episodes == 0 and episode!=0:
        #     log_all(high_level_controller, path, q_loss_history, policy_loss_history, rew_history, episode, save_network='high_level')
        #     log_all(step_predictor, path, low_q_loss_history, low_policy_loss_history, low_rew_history, episode,  additional_plot=dict(data=step_history, name='step_plot'), save_network='low_level')

        



    return l_vars, labels

    # TODO: plot latent space
        

if __name__ == "__main__":
    # TODO: Do with json load for future
    # from experiments_configs.half_cheetah_multi_env import config as env_config

    inference_path = '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_06_06_00_56_default_true_gmm'

    
    # complex_agent_config = dict(
    #     environment = HalfCheetahMixtureEnv(env_config),
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'half_cheetah_definitive_training',
    #     epoch = 1000,
    # )
    # complex_agent_position = dict(
    #     environment = HalfCheetahMixtureEnv(env_config),
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'half_cheetah_dt0.01_mall_vel3',
    #     epoch = 6000,
    # )
    # complex_agent_vel = dict(
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'half_cheetah_dt0.01_only_vel',
    #     epoch = 2000,
    # )
    complex_agent = dict(
        environment = HopperMulti,
        experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
        experiment_name = 'hopper_26_06',
        epoch = 500,
    )
    # complex_agent_vel = dict(
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'walker_curriculum',
    #     epoch = 1500,
    # )
    

    with open(complex_agent['experiments_repo'] + complex_agent['experiment_name'] + '/config.json', 'r') as file:
        env_config = json.load(file)

    # complex_agent_config = dict(
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'hopper_change_task_dt_0.01(before0.004)',
    #     epoch = 16000,
    # )
    # with open(complex_agent_config['experiments_repo'] + complex_agent_config['experiment_name'] + '/config.json', 'r') as file:
    #     env_config = json.load(file)
    # environment = HopperMulti(env_config)
    # complex_agent_config['environment'] = environment

    env = complex_agent['environment'](env_config)
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

    memory = Memory(1e+6)
    encoder = get_encoder(inference_path, shared_dim, encoder_input_dim)
    transfer_function = get_complex_agent(env, complex_agent)

    rollout(env,transfer_function, variant, obs_dim, max_path_len, n_tasks=1, save_video_path=inference_path)