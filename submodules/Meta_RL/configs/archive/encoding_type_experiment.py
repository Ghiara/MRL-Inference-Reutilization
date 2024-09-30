import copy
from collections import OrderedDict
import torch

from smrl.utility.ops import deep_dictionary_update

from smrl.vae.encoder_networks import GRUEncoder
from smrl.data_management.replay_buffers import MultiTaskReplayBuffer
from smrl.vae.decoder_networks import SeparateMlpDecoder
from smrl.vae.mdpvae import NeuralProcess, MdpVAE
from rlkit.torch.networks import ConcatMlp
from smrl.policies.meta_policy import MetaRLTanhGaussianPolicy

base_config = OrderedDict(
    description = {
        'name': 'Buffer-experiment-base-config',
        'file': __file__,
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 1,
    context_size = 5,

    # Inference mechanism
    encoder_type = GRUEncoder,
    encoder_kwargs = dict(
        hidden_size=32,
        num_layers=4,
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
    qf_network_type = ConcatMlp,
    qf_network_kwargs = {
        'init_w': 1.0,
        'hidden_sizes': [16, 16, 16,],
    },
    policy_type = MetaRLTanhGaussianPolicy,
    policy_kwargs = {
        'hidden_sizes': [16, 16, 16,],
        'std': None,
        'init_w': 1.0,
    },

    # Buffer
    replay_buffer_type = MultiTaskReplayBuffer,
    replay_buffer_kwargs = {
        'max_replay_buffer_size': 50_000,
        'max_sub_size': 2_500,
        'randomize_targets': True,
        'randomize_contexts': False,
    },

    # Algorithm
    path_collector_kwargs = {},
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
        num_epochs=5_000,             # Number of training epochs
        max_path_length=250,        # Maximum path length
        prediction_target_size=5,  # Size of the prediction targets for ELBO computation (number of transitions which the decoder needs to predict (reward & next observation))
        num_eval_paths_per_epoch=25,  # Number of transitions used for evaluation
        num_expl_paths_per_epoch=10, # Number of transitions added to the replay buffer in each train loop
        num_inference_trains_per_train_loop=5,  # Number of inference training steps in each train loop
        num_policy_trains_per_train_loop=5, # Number of policy training steps in each train loop
        num_train_loops_per_epoch=20,    # Number of train loops per epoch
    ),
)

experiment1 = dict(
    description = {
        'name': 'encodings-experiment-1',
        'file': __file__,
        'variant': "sample"
    },
    encoder_kwargs = dict(
        encoding_mode='sample',
    ),
)
experiment1 = deep_dictionary_update(copy.deepcopy(base_config), experiment1)

experiment2 = OrderedDict(
    description = {
        'name': 'encodings-experiment-2',
        'file': __file__,
        'variant': "mean"
    },
    encoder_kwargs = dict(
        encoding_mode='mean',
    ),
)
experiment2 = deep_dictionary_update(copy.deepcopy(base_config), experiment2)

experiment3 = OrderedDict(
    description = {
        'name': 'encodings-experiment-3',
        'file': __file__,
        'variant': "mean_var"
    },
    encoding_dim = 2,
    encoder_kwargs = dict(
        encoding_mode='mean_var',
    ),
)
experiment3 = deep_dictionary_update(copy.deepcopy(base_config), experiment3)