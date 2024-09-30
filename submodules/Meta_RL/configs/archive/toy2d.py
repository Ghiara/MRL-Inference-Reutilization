import torch
import copy
from collections import OrderedDict

from rlkit.torch.networks import ConcatMlp
from smrl.vae.encoder_networks import GRUEncoder
from smrl.vae.decoder_networks import SeparateMlpDecoder
from smrl.vae.mdpvae import NeuralProcess
from smrl.policies.meta_policy import MetaRLTanhGaussianPolicy
from smrl.policies.exploration import RandomMemoryPolicy
from smrl.data_management.replay_buffers import MultiTaskReplayBuffer

from smrl.utility.ops import deep_dictionary_update


config = OrderedDict(

    description = {
        'name': 'Toy2d-config',
        'file': __file__,
        'variant': 'Toy2d-config',
        'inference': 'Neural process + GRU',
    },

    observation_dim = 2,
    action_dim = 2,
    latent_dim = 2,
    encoding_dim = 2,   # Can be different to latent_dim, e.g. if encoder.encoding_mode is equal to 'mean_var'
    context_size = 5,

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
    qf_network_type = ConcatMlp,
    qf_network_kwargs = {
        'init_w': 1.0,
        'hidden_sizes': [32, 32, 32, 32],
    },
    policy_type = MetaRLTanhGaussianPolicy,
    policy_kwargs = {
        'hidden_sizes': [32, 32, 32, 32],
        'std': None,
        'init_w': 1.0,
    },

    # Algorithm
    path_collector_kwargs = {
        'save_env_in_snapshot': True,
    },
    replay_buffer_type = MultiTaskReplayBuffer,
    replay_buffer_kwargs = {
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
        num_epochs=10_000,             # Number of training epochs
        max_path_length=250,        # Maximum path length
        prediction_target_size=5,  # Size of the prediction targets for ELBO computation (number of transitions which the decoder needs to predict (reward & next observation))
        num_eval_paths_per_epoch=25,  # Number of transitions used for evaluation
        num_expl_paths_per_epoch=10, # Number of transitions added to the replay buffer in each train loop
        num_inference_trains_per_train_loop=5,  # Number of inference training steps in each train loop
        num_policy_trains_per_train_loop=5, # Number of policy training steps in each train loop
        num_train_loops_per_epoch=20,    # Number of train loops per epoch
    ),
)

off_policy_config = dict(
    description = {
        'name': 'Toy2d-config-offpolicy',
        'file': __file__,
        'variant': 'Toy2d-config-offpolicy',
        'inference': 'Neural process + GRU',
    },
    inference_policy_type = RandomMemoryPolicy,   
    inference_policy_kwargs = {
        'action_update_interval': 25,
        'std': 0.1,
    },
)
off_policy_config = deep_dictionary_update(copy.deepcopy(config), off_policy_config)
