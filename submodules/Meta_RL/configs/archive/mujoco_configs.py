import torch
import copy
from collections import OrderedDict

from smrl.vae.encoder_networks import GRUEncoder
from smrl.vae.decoder_networks import SeparateMlpDecoder
from smrl.vae.mdpvae import NeuralProcess
from smrl.policies.meta_policy import MetaRLTanhGaussianPolicy
from smrl.policies.meta_value_function import MlpValueFunction
from smrl.data_management.replay_buffers import MultiTaskReplayBuffer, TrajectoryReplayBuffer

from smrl.utility.ops import deep_dictionary_update


base_config = OrderedDict(

    description = {
        'name': 'MuJoCo-Config',
        'file': __file__,
        'variant': 'MuJoCo config',
    },

    observation_dim = None,
    action_dim = None,
    latent_dim = None,
    encoding_dim = None,
    context_size = 50,

    # Inference mechanism
    encoder_type = GRUEncoder,
    encoder_kwargs = dict(
        hidden_size=64,
        num_layers=4,
        encoding_mode='mean_var',
    ),
    decoder_type = SeparateMlpDecoder,
    decoder_kwargs = {
        'hidden_sizes': [16,16,16],
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
        'hidden_sizes': [64, 64, 64, 64, 64],
    },
    policy_type = MetaRLTanhGaussianPolicy,
    policy_kwargs = {
        'hidden_sizes': [64, 64, 64, 64],
    },

    # Algorithm
    path_collector_kwargs = {},
    replay_buffer_type = TrajectoryReplayBuffer,
    replay_buffer_kwargs = {
        'max_path_number': 500,
        'max_sub_size': 2_500,
        'randomize_targets': True,
        'randomize_contexts': False,
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
        batch_size=256,             # Batch size for batch-training (= number of samples which is passed to the trainers in each training step)
        num_epochs=25_000,             # Number of training epochs
        max_path_length=250,        # Maximum path length
        prediction_target_size=5,   # Size of the prediction targets for ELBO computation (number of transitions which the decoder needs to predict (reward & next observation))
        num_eval_paths_per_epoch=15,  # Number of transitions used for evaluation
        num_expl_paths_per_epoch=10, # Number of transitions added to the replay buffer in each train loop
        num_inference_paths_per_epoch=10,
        num_inference_trains_per_train_loop=5,  # Number of inference training steps in each train loop
        num_policy_trains_per_train_loop=5, # Number of policy training steps in each train loop
        num_train_loops_per_epoch=20,    # Number of train loops per epoch
    ),
) 


cheetah_config = deep_dictionary_update(
    copy.deepcopy(base_config),
    dict(
        observation_dim = 17,
        action_dim = 6,
        latent_dim = 1,
        encoding_dim = 2,
    )
)

ant_config = deep_dictionary_update(
    copy.deepcopy(base_config),
    dict(
        observation_dim = 27,
        action_dim = 8,
        latent_dim = 1,
        encoding_dim = 2,
    )
)
