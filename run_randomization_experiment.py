"""
This file serves as an example how to instantiate an algorithm instance
from a configuration file and run it afterwards.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-04-06
"""

import os
import torch
import ray
from datetime import datetime
import pytz

from smrl.experiment.experiment_setup import setup_experiment
from configs.base_configuration import config
from configs.environment_factory import toy1d_rand, toy1d_domain_rand
import torch
from smrl.vae.encoder_networks import GRUEncoder, MlpEncoder
from smrl.vae.decoder_networks import MlpDecoder, SeparateMlpDecoder
from smrl.vae.mdpvae import NeuralProcess, MdpVAE, InfoMaxMdpVAE
from smrl.policies.meta_policy import MetaRLTanhGaussianPolicy
from smrl.policies.meta_value_function import MlpValueFunction
from smrl.policies.exploration import RandomPolicy, RandomMemoryPolicy
from smrl.data_management.replay_buffers import ContextReplayBuffer, TrajectoryReplayBuffer, MultiTaskReplayBuffer
from collections import OrderedDict

config = OrderedDict(

    description = {
        'name': 'Domain_randomization_confi',
        'file': __file__,
        'variant': 'Base configuration',
    },
    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 1,   # Can be different to latent_dim, e.g. if encoder.encoding_mode is equal to 'mean_var'
    context_size = 5,
    multipliers = [2,0.5],
    # Inference mechanism
    encoder_type = GRUEncoder,
    encoder_kwargs = dict(
        hidden_size=32,
        num_layers=4,
        encoding_mode='sample',
    ),
    decoder_type = SeparateMlpDecoder,
    decoder_kwargs = {
        'hidden_sizes': [8, 8,],
        'std_rew': 0.1,
        'std_obs': 0.1,
        'activation_function': torch.nn.ReLU,
    },
    inference_network_type = NeuralProcess,
    inference_network_kwargs = {
        'beta': 1.0,
    },

    # Policy training
    qf_network_type = MlpValueFunction,
    qf_network_kwargs = {
        'hidden_sizes': [16, 16, 16,],
    },
    policy_type = MetaRLTanhGaussianPolicy,
    policy_kwargs = {
        'hidden_sizes': [16, 16, 16,],
    },

    # Exploration / rollouts
    # expl_policy_type = RandomMemoryPolicy,
    # expl_policy_kwargs = {
    #     'action_update_interval': 10,
    #     'std': 0.05,
    # },
    # inference_policy_type = RandomMemoryPolicy,    # Exploration policy for off-policy inference training (None -> on-policy training)
    # inference_policy_kwargs = {
    #     'action_update_interval': 25,
    #     'mean_std': 0.05,
    #     'std_mean': 0.025,
    # },    # Arguments for exploration policy (inference training) (off-policy training)


    # Algorithm
    path_collector_kwargs = {},
    replay_buffer_type = TrajectoryReplayBuffer,
    replay_buffer_kwargs = {
        'max_path_number': 1_000,
        'max_sub_size': 300,
    },
    inference_replay_buffer_type = MultiTaskReplayBuffer,
    inference_replay_buffer_kwargs = {
        'max_replay_buffer_size': 50_000,
        'max_sub_size': 2_500,
        'randomize_targets': True,
        'randomize_contexts': False,
    },
    inference_trainer_kwargs = dict(
        lr=3e-4,                    # Learning rate for the decoder & encoder networks
        n_latent_samples=32,        # Number of latent samples for Monte-Carlo estimation
        clipping=None,
    ),
    policy_trainer_kwargs = dict(
        discount=0.99,              # Discount factor of the MDP
        policy_lr=1e-4,             # Learning rate for the policy network
        qf_lr=1e-4,                 # Learning rate for the Q-function networks
        # clipping=1e2,               
        encoder_lr=None,
        use_automatic_entropy_tuning=False,
    ),
    algorithm_kwargs = dict(
        batch_size=128,             # Batch size for batch-training (= number of samples which is passed to the trainers in each training step)
        num_epochs=2_000,             # Number of training epochs
        max_path_length=250,        # Maximum path length
        prediction_target_size=5,  # Size of the prediction targets for ELBO computation (number of transitions which the decoder needs to predict (reward & next observation))
        num_eval_paths_per_epoch=15,    # Number of trajectories used for evaluation
        num_expl_paths_per_epoch=10,    # Number of trajectories added to the policy replay buffer in each train loop
        num_inference_paths_per_epoch=10,   # Number of trajectories added to the inference replay buffer in each train loop
        num_inference_trains_per_train_loop=5,  # Number of inference training steps in each train loop
        num_policy_trains_per_train_loop=5, # Number of policy training steps in each train loop
        num_train_loops_per_epoch=20,    # Number of train loops per epoch
    ),
)
# Environment
config['environment_factory'] = toy1d_domain_rand


# GPU available?
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    # Makes sure the listing is in the same order as with nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"    # Set to "" if you do not want to use GPUs
print(f"GPU available: {'Yes' if torch.cuda.is_available() else 'No'}")

# Multithreading
os.environ["MULTITHREADING"] = "True"   # Set to "False" to not use multithreading
if os.environ["MULTITHREADING"] == "True":
    ray.init(num_cpus=12)


# Setup experiment, modify logging parameters

# multipliers = [[5,0.3], [10,0.2], [1.5, 0.8]]
# multipliers = [[2,0.5]]
multipliers = [[1.1,0.9]]
for mult in multipliers:
    experiment_name = "randomization_experiment_randn"
    current_time = datetime.now().astimezone(pytz.timezone('Europe/Berlin'))
    logger_kwargs = {
        'log_dir': f"./data/{experiment_name}_mult:{mult[1]}-{mult[0]}(toy1d_rand)",
        # './data/delete'
        'snapshot_mode': 'gap_and_last',
        'snapshot_gap': 10,
    }
    config['multipliers'] = mult
    print("MULTIPLIERS: ", mult[0], mult[1])
    algorithm, description = setup_experiment(
        experiment_name, 
        config,
        logger_kwargs=logger_kwargs
    )
    # RUN EXPERIMENT
    algorithm.train()