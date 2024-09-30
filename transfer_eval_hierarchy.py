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

from experiments_configs.half_cheetah_multi_env import config as env_config

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
ptu.set_gpu_mode(False)
frames = []
image_info = dict(reward = [],
obs = [],
base_task = [])

# TODO: get velocities for all agents

# TODO: einheitliches set to device
simple_env_dt = 0.05
sim_time_steps = 1
toy_steps = 1
max_path_len=10000

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
    rewards: torch.Tensor, 
    next_observations: torch.Tensor,
    action,
    step='set_position'):
    """
    Maps transitions from the cheetah environment
    to the discrete, one-dimensional goal environment.
    """
    velocity_x = 3
    simple_env_dt = 0.05
    ### little help: [0:3] gives elements in positions 0,1,2 
    simple_observations = np.zeros(obs_dim)
    simple_observations[...,0] = observations[...,-3]
    simple_observations[...,1] = observations[...,8]
    # simple_observations[...,1:3] = observations[...,0:2]
    # simple_observations[...,4:] = observations[...,8:10]          # TODO: check if this is the velocity
    next_simple_observations = np.zeros(obs_dim)
    next_simple_observations[...,0] = next_observations[...,-3]
    next_simple_observations[...,1] = next_observations[...,8]
    # next_simple_observations[...,1:3] = next_observations[...,0:2]
    # next_simple_observations[...,3:] = next_observations[...,8:11]
    # simple_actions = next_simple_observations[:3] - simple_observations[:3]
    # # simple_actions = next_simple_observations[[0,4,2,3,5]]
    # simple_action = np.zeros_like(next_simple_observations)
    # simple_action = np.zeros(1)
    # # place h9lder, can be removed
    # # simple_action[1] = action[1]
    # # simple_action[0] = next_simple_observations[1]/velocity_x
    # simple_action[0] = (next_simple_observations[0] - simple_observations[0])/(velocity_x * simple_env_dt * sim_time_steps)
    simple_action = next_simple_observations[1]/1.2

    return simple_action, rewards, next_simple_observations

def cheetah_to_simple_env_obs(obs):
    simple_observations = np.zeros(obs_dim)
    simple_observations[...,0] = obs[...,-3]
    # simple_observations[...,1:3] = obs[...,1:3]
    # simple_observations[...,3:] = obs[...,7:10]
    simple_observations[...,1] = obs[...,8]
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
            cv2.putText(frame, 'action: ' + str(info['obs'][i]), (0, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            # Apply transformation if any
            if transform is not None:
                frame = transform(frame)
            else:
                # Convert color space if no transformation provided
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            
            writer.append_data(frame)

            cv2.imwrite('/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_10_13_56_50_default_true_gmm/frame.png', frame)

def get_decoder(path, action_dim, obs_dim, reward_dim, latent_dim, output_action_dim, net_complex_enc_dec, variant):
    path = os.path.join(path, 'weights')
    for filename in os.listdir(path):
        if filename.startswith('decoder'):
            name = os.path.join(path, filename)
        if filename.startswith('retrained_decoder'):
            name = os.path.join(path, filename)
            break
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

def create_tsne(latent_variables, task_labels, path):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    save_as = os.path.join(path , 'tsne_test.png')
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

def step_cheetah(task, obs, transfer_function):

    for i in range(sim_time_steps):
            
            complex_action = transfer_function.get_action(ptu.from_numpy(obs), task, return_dist=False)
            next_obs, r, internal_done, truncated, env_info = env.step(complex_action.detach().cpu().numpy())

            image = env.render()
            frames.append(image)
            image_info['reward'].append(r)
            image_info['obs'].append(task)
            image_info['base_task'].append(env.task)
            # if internal_done:
            #     break

            obs = next_obs
        
    return r, next_obs

def step_toy(simple_obs, mu):
    for _ in range(toy_steps):
        simple_env.sim.data.set_joint_qpos('boxslideX', simple_obs[0])
        simple_env.sim.data.set_joint_qvel('boxslideX', simple_obs[1])

        # Save latent vars
        policy_input = torch.cat([ptu.from_numpy(simple_obs), mu.squeeze()], dim=-1)
        simple_action, info = (simple_agent.get_torch_actions(policy_input, deterministic=True), [{}] * simple_obs.shape[0])

        _,_, logits = decoder(ptu.from_numpy(simple_obs), simple_action, 0, mu.squeeze()) #latent_variables.unsqueeze(1).repeat(1, states.shape[1], 1)

        simple_obs,_,_,_ = simple_env.step(simple_action.detach().cpu().numpy())

    simple_action = torch.from_numpy(simple_obs)

    return simple_action, logits

def update_task(env, task):
    if task[0]<0:
        env.base_task = 5
        env.task = task
    elif task[0]>0:
        env.base_task = 4
        env.task = task
    elif task[3]<0:
        env.base_task = 1
        env.task = np.array([task[3], 0, 0, 0, 0])
    elif task[3]>0:
        env.base_task = 0
        env.task = np.array([task[3], 0, 0, 0, 0])
def rollout(task, env, encoder, decoder, simple_agent, transfer_function_pos, transfer_function_vel, 
            variant, obs_dim, actions_dim, max_path_len, 
            n_tasks, inner_loop_steps, save_video_path):
    range_dict = OrderedDict(pos_x = [0.5, 25],
                             velocity_z = [1.5, 3.],
                             pos_y = [np.pi / 6., np.pi / 2.],
                             velocity_x = [0.5, 3.0],
                             velocity_y = [2. * np.pi, 4. * np.pi],
                             )

    path_length = 0
    obs = env.reset()[0]
    contexts = torch.zeros((n_tasks, variant['algo_params']['time_steps'], obs_dim + 1 + obs_dim), device=DEVICE)

    update_task(env, task)
    l_vars = []
    labels = []
    x_pos_plot = []
    x_vel_plot = []
    for path_length in range(max_path_len):

        # get encodings
        encoder_input = contexts.detach().clone()
        encoder_input = encoder_input.view(encoder_input.shape[0], -1).to(DEVICE)
        mu, log_var = encoder(encoder_input)     # Is this correct??

        # Save values for plotting
        if env.task[0]<0:
            label = -1
        elif env.task[0]>0:
            label = 1
        elif env.task[3]<0:
            label = -2
        elif env.task[3]>0:
            label = 2
        if path_length == 0:
            l_vars = mu.detach().cpu().numpy()
            labels = np.array([label])
        else:
            l_vars = np.concatenate((l_vars, mu.detach().cpu().numpy()), axis = 0)
            labels = np.concatenate((labels, np.array([label])), axis = 0)

        # Get simple action
        complex_obs_before = np.copy(obs)
        simple_obs = cheetah_to_simple_env_obs(obs)
        obs_before_sim = np.copy(simple_obs)

        simple_action, logits = step_toy(simple_obs, mu)

        task_prediction = torch.argmax(torch.nn.functional.softmax(logits), dim=0)
        # task_prediction = 2

        # Convert to action the cheetah can use
        # scaled_simple_action = scale_simple_action(simple_action, simple_obs)
        scaled_simple_action = simple_action
        if env_config['task_dim'] == 5:
            scaled_simple_action = torch.Tensor([scaled_simple_action[0], 0,0,scaled_simple_action[1],0])
        if env_config['task_dim'] == 2:
            scaled_simple_action = torch.Tensor([scaled_simple_action[0],scaled_simple_action[1]])
        
        # task_prediction = 2
        if env_config['task_dim'] == 5:
            if task_prediction == 0 or task_prediction == 1:    # velocity
                scaled_simple_action[0] = 0
            elif task_prediction == 2 or task_prediction == 3:  # position
                scaled_simple_action[3] = 0
        elif env_config['task_dim'] == 2:
            if task_prediction == 0 or task_prediction == 1:    # velocity
                scaled_simple_action[0] = 0
            elif task_prediction == 2 or task_prediction == 3:  # position
                scaled_simple_action[1] = 0
                
        print('INPUT:', scaled_simple_action)

        # Make cheetah step
        obs = env._get_obs()
        if task_prediction == 2 or task_prediction == 3:
            r, next_obs = step_cheetah(scaled_simple_action, obs, transfer_function=transfer_function_pos)
        elif task_prediction == 0 or task_prediction == 1:
            r, next_obs = step_cheetah(scaled_simple_action, obs, transfer_function=transfer_function_vel)
        
        map_state = cheetah_to_simple_env_map(complex_obs_before, r, next_obs, simple_action)
        print(map_state[2])
        x_pos_plot.append(map_state[2][0])
        x_vel_plot.append(map_state[2][1])

        # xpos_before = simple_obs[0]
        # xpos_after = map_state[2][0]
        # xvel = map_state[2][1]
        # avg_vel = (xpos_after - obs_before_sim[0])/(sim_time_steps * simple_env_dt)
        # map_state[2][1] = avg_vel

        # if env.task[0] != 0:
        #     # reward = - np.abs(xpos_after - env.task[0]) / np.abs(env.task[0])

        # elif env.task[3] != 0:
        #     reward = -1.0 * np.abs(avg_vel - env.task[3]) / np.abs(env.task[3])
        print(env.task, r)
        data = torch.cat([ptu.from_numpy(obs_before_sim), torch.unsqueeze(torch.tensor(r, device=DEVICE), dim=0), ptu.from_numpy(map_state[2])], dim=0).unsqueeze(dim=0)
        context = torch.cat([contexts.squeeze(), data], dim=0)
        contexts = context[-time_steps:, :]
        contexts = contexts.unsqueeze(0).to(torch.float32)


    size = frames[0].shape
    window_size = 10

    def moving_average(data, window_size):
        # """Compute the moving average using a sliding window."""
        # window = np.ones(int(window_size))/float(window_size)
        # return np.convolve(data, window, 'same')
        from scipy.ndimage.filters import gaussian_filter1d
        return gaussian_filter1d(data, sigma=2)
    

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x_pos_plot = moving_average(x_pos_plot, window_size)
    # Plot position on the first (left) axis
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)', color=color)
    ax1.plot(np.arange(max_path_len), np.array(x_pos_plot), label='Position', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    if task[0]!=0:
        ax1.plot(np.arange(max_path_len), np.ones(max_path_len)*task[0], linestyle='--', color=color)

    # Create a second axis sharing the same x-axis
    x_vel_plot = moving_average(x_vel_plot, window_size)
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Velocity (m/s)', color=color)
    ax2.plot(np.arange(max_path_len), np.array(x_vel_plot), label='Velocity', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    if task[3]!=0:
        ax2.plot(np.arange(max_path_len), np.ones(max_path_len)*task[3], linestyle='--', color=color)

    # Save the figure to a file
    filename = os.path.join(save_video_path, f"{task}.png")
    plt.savefig(filename, dpi=300)  # Save the figure with 300 dpi resolution
    plt.close()

    # Save to corresponding repo
    fps=10
    save_as = f'{save_video_path}/videos/transfer.gif'
    video = cv2.VideoWriter(save_as, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    # Write frames to video
    _frames_to_gif(video, frames, image_info, save_as)
    video.release()




    return l_vars, labels

    # TODO: plot latent space
        

if __name__ == "__main__":

    # inference_path = '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_06_15_05_36_default_true_gmm'
    # inference_path = '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_08_14_42_39_default_true_gmm'
    # inference_path = '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_17_20_51_34_default_true_gmm'
    inference_path = '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_15_10_43_10_default_true_gmm'
    
    complex_agent_position = dict(
        environment = HalfCheetahMixtureEnv(env_config),
        experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
        experiment_name = 'half_cheetah_dt0.01_mall_vel3',
        epoch = 6000,
    )
    complex_agent_vel = dict(
        experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
        experiment_name = 'half_cheetah_dt0.01_only_vel',
        epoch = 2000,
    )

    env = complex_agent_position['environment']
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

    encoder = get_encoder(inference_path, shared_dim, encoder_input_dim)
    simple_agent = get_simple_agent(inference_path, obs_dim, policy_latent_dim, action_dim, m)
    transfer_function_pos = get_complex_agent(env, complex_agent_position)
    transfer_function_vel = get_complex_agent(env, complex_agent_vel)
    output_action_dim = 8
    decoder = get_decoder(inference_path, action_dim, obs_dim, reward_dim, latent_dim, output_action_dim, net_complex_enc_dec, variant)

    ### ROLLOUT ###
    tasks = [
        np.array([5.0,0,0,0,0]),
        np.array([-5.0,0,0,0,0]),
        np.array([0,0,0,2.0,0]), 
        np.array([0,0,0,-2,0])
        ]
    for i, task in enumerate(tasks):
        if i == 0:
            res = rollout(task, env, encoder, decoder, simple_agent, 
                                        transfer_function_pos, transfer_function_vel, variant, obs_dim, action_dim, 
                                        max_path_len, n_tasks=1, inner_loop_steps=10, save_video_path=inference_path)
            latent_vars = res[0]
            labels = res[1]
        else:
            res = rollout(task, env, encoder, decoder, simple_agent, 
                                        transfer_function_pos, transfer_function_vel, variant, obs_dim, action_dim, 
                                        max_path_len, n_tasks=1, inner_loop_steps=10, save_video_path=inference_path)
            latent_vars = np.concatenate((latent_vars, res[0]), axis = 0)
            labels = np.concatenate((labels, res[1]), axis=0)

    ### Save metadata, tensors, videos
    create_tsne(np.array(latent_vars), np.array(labels), inference_path)