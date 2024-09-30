import torch
import copy
from collections import OrderedDict

from smrl.vae.encoder_networks import GRUEncoder
from smrl.vae.decoder_networks import SeparateMlpDecoder
from smrl.vae.mdpvae import NeuralProcess
from smrl.policies.meta_policy import MetaRLTanhGaussianPolicy
from smrl.policies.exploration import RandomMemoryPolicy
from smrl.policies.meta_value_function import MlpValueFunction
from smrl.data_management.replay_buffers import MultiTaskReplayBuffer, TrajectoryReplayBuffer

from smrl.utility.ops import deep_dictionary_update



standard_qfunction = OrderedDict(

    description = {
        'name': 'standard-qfunction',
        'file': __file__,
        'variant': 'standard q-function',
    },

    observation_dim = 2,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 25,

    # Inference mechanism
    encoder_type = GRUEncoder,
    encoder_kwargs = dict(
        hidden_size=32,
        num_layers=4,
        encoding_mode='mean_var',
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
        'hidden_sizes': [32, 32, 32],
    },
    policy_type = MetaRLTanhGaussianPolicy,
    policy_kwargs = {
        'hidden_sizes': [32, 32, 32],
    },

    # Off-policy inference training
    inference_policy_type = RandomMemoryPolicy,
    inference_policy_kwargs = {
        'action_update_interval': 50,
        'mean_std': 0.05,
        'std_mean': 0.025,
    },

    # Algorithm
    path_collector_kwargs = {},
    replay_buffer_type = TrajectoryReplayBuffer,
    replay_buffer_kwargs = {
        'max_replay_buffer_size': 50_000,
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
        encoder_lr=None,
        use_automatic_entropy_tuning=False,
    ),
    algorithm_kwargs = dict(
        batch_size=128,             # Batch size for batch-training (= number of samples which is passed to the trainers in each training step)
        num_epochs=5_000,             # Number of training epochs
        max_path_length=250,        # Maximum path length
        prediction_target_size=5,  # Size of the prediction targets for ELBO computation (number of transitions which the decoder needs to predict (reward & next observation))
        num_eval_paths_per_epoch=15,  # Number of transitions used for evaluation
        num_expl_paths_per_epoch=10, # Number of transitions added to the replay buffer in each train loop
        num_inference_trains_per_train_loop=5,  # Number of inference training steps in each train loop
        num_policy_trains_per_train_loop=5, # Number of policy training steps in each train loop
        num_train_loops_per_epoch=20,    # Number of train loops per epoch
    ),
) 



qf_lr_small = dict(
    description = {
        'name': 'qf-lr-1e-5',
        'file': __file__,
        'variant': 'qf_LR = 5e-6',
    },
    policy_trainer_kwargs = dict(
        qf_lr=5e-6,
    ),
)
qf_lr_small = deep_dictionary_update(copy.deepcopy(standard_qfunction), qf_lr_small)

qf_lr_high = dict(
    description = {
        'name': 'qf-lr-5e-3',
        'file': __file__,
        'variant': 'qf_LR = 5e-3',
    },
    policy_trainer_kwargs = dict(
        qf_lr=5e-3,
    ),
)
qf_lr_high = deep_dictionary_update(copy.deepcopy(standard_qfunction), qf_lr_high)

simple_q_function = dict(
    description = {
        'name': 'simple-qfunction',
        'file': __file__,
        'variant': 'hidden_sizes = [16,16]',
    },
    qf_network_kwargs = {
        'hidden_sizes': [16, 16],
    },
)
simple_q_function = deep_dictionary_update(copy.deepcopy(standard_qfunction), simple_q_function)

complex_q_function = dict(
    description = {
        'name': 'complex-qfunction',
        'file': __file__,
        'variant': 'hidden_sizes = [64,64,64,64]',
    },
    qf_network_kwargs = {
        'hidden_sizes': [64,64,64,64],
    },
)
complex_q_function = deep_dictionary_update(copy.deepcopy(standard_qfunction), complex_q_function)
