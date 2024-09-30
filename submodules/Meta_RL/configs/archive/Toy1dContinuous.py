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
from smrl.vae.encoder_decorators.striding import StridedEncoder

from smrl.utility.ops import deep_dictionary_update

from specific.encoder_decorators import Toy1dForToy1dContinuousDecorator
from specific.encoders import Toy1dOracle


base_config = OrderedDict(

    description = {
        'name': 'Continuous-config',
        'file': __file__,
        'variant': 'Continuous-config',
    },

    observation_dim = 2,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 1,   # Can be different to latent_dim, e.g. if encoder.encoding_mode is equal to 'mean_var'
    context_size = 25,

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
        'hidden_sizes': [32, 32, 32],
    },
    policy_type = MetaRLTanhGaussianPolicy,
    policy_kwargs = {
        'hidden_sizes': [32, 32, 32],
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
        lr=3e-3,                    # Learning rate for the decoder & encoder networks
        n_latent_samples=32,        # Number of latent samples for Monte-Carlo estimation
        clipping=None,
    ),
    policy_trainer_kwargs = dict(
        discount=0.99,              # Discount factor of the MDP
        policy_lr=1e-3,             # Learning rate for the policy network
        qf_lr=1e-3,                 # Learning rate for the Q-function networks
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
        'name': 'Continuous-config-offpolicy',
        'file': __file__,
        'variant': 'Continuous-config-offpolicy',
    },
    encoding_dim = 2,
    inference_policy_type = RandomMemoryPolicy,
    inference_policy_kwargs = {
        'action_update_interval': 25,
        'mean_std': 0.075,
        'std_mean': 0.05,
    },
    encoder_kwargs = {
        'encoding_mode': 'mean_var',
    }
)
off_policy_config = deep_dictionary_update(copy.deepcopy(base_config), off_policy_config)


strided_config = dict(
    description = {
        'name': 'Continuous-config-strided',
        'file': __file__,
        'variant': 'Continuous-config-strided',
    },
    context_size = 25,
    encoder_decorator_type = StridedEncoder,
    encoder_decorator_kwargs = {
        'stride': 5,
    },
    encoder_kwargs = {
        "context_size": 5,
    }
)
strided_config = deep_dictionary_update(copy.deepcopy(base_config), strided_config)


encoder_transfer_config = dict(

    description = {
        'name': 'Toy1d-transfer',
        'file': __file__,
        'variant': 'Transfer from Toy1D to Toy1dDiscretized without training',
    },

    # TODO: striding
    encoder_decorator_type = Toy1dForToy1dContinuousDecorator,
    encoder_decorator_kwargs = {
        "path_to_weights": '/data/bing/julius/Toy1D_Buffer-experiment-3_2023-01-10_11-36-11/params.pkl',
        "trainable": False,
    },
    encoder_type=GRUEncoder,
    encoder_kwargs = dict(
        hidden_size=32,
        num_layers=4,
        encoding_mode='sample',
    ),
)
encoder_transfer_config = deep_dictionary_update(copy.deepcopy(base_config), encoder_transfer_config)


oracle_config = dict(
    description = {
        'name': 'Toy1d-oracle',
        'file': __file__,
        'variant': 'Oracle encoder',
    },
    encoder_type = Toy1dOracle,
    context_size = 5,
)
oracle_config = deep_dictionary_update(copy.deepcopy(base_config), oracle_config)

powerful_qfunction = dict(
    description = {
        'name': 'Off-policy-Powerful-qfunction',
        'file': __file__,
        'variant': 'Powerful q-function (off-policy, mean-var)',
    },
    qf_network_kwargs = {
        'hidden_sizes': [128, 128, 128, 128, 128, 128,],
    },
)
powerful_qfunction = deep_dictionary_update(copy.deepcopy(off_policy_config), powerful_qfunction)
