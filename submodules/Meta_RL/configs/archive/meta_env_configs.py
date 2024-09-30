import torch
import copy
from smrl.vae.encoder_networks import GRUEncoder
from smrl.vae.decoder_networks import SeparateMlpDecoder
from smrl.vae.mdpvae import NeuralProcess
from smrl.policies.meta_policy import MetaRLTanhGaussianPolicy
from smrl.policies.meta_value_function import MlpValueFunction
from smrl.data_management.replay_buffers import TrajectoryReplayBuffer, MultiTaskReplayBuffer
from collections import OrderedDict

base_config = OrderedDict(

    description = {
        'name': 'Mujoco-Envs-Base-config',
        'file': __file__,
        'variant': 'Mujoco envi. base configuration',
    },

    observation_dim = None,
    action_dim = None,
    latent_dim = None,
    encoding_dim = None,
    context_size = None,

    # Inference mechanism
    encoder_type = GRUEncoder,
    encoder_kwargs = dict(
        hidden_size=128,
        num_layers=2,
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
        'hidden_sizes': [64,64,64,64,64],
    },
    policy_type = MetaRLTanhGaussianPolicy,
    policy_kwargs = {
        'hidden_sizes': [64,64,64,64,64],
    },

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
        lr=3e-4,
        n_latent_samples=32,
        clipping=None,
    ),
    policy_trainer_kwargs = dict(
        discount=0.99,
        policy_lr=1e-4,
        qf_lr=1e-4,
        use_automatic_entropy_tuning=False,
    ),
    algorithm_kwargs = dict(
        batch_size=128,
        num_epochs=5_000,
        max_path_length=250,
        prediction_target_size=5,
        num_eval_paths_per_epoch=15,
        num_expl_paths_per_epoch=10,
        num_inference_paths_per_epoch=10,
        num_inference_trains_per_train_loop=5,
        num_policy_trains_per_train_loop=5,
        num_train_loops_per_epoch=20,
    ),
)


half_cheetah_config = copy.deepcopy(base_config)
half_cheetah_config.update(dict(
    description = {
        'name': 'Half-cheetah-config',
        'file': __file__,
        'variant': 'Base configuration',
    },

    observation_dim = 20,
    action_dim = 6,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 10,
))

ant_config = copy.deepcopy(base_config)
ant_config.update(dict(
    description = {
        'name': 'Ant-config',
        'file': __file__,
        'variant': 'Base configuration',
    },

    observation_dim = 116,
    action_dim = 8,
    latent_dim = 2,
    encoding_dim = 4,
    context_size = 25,
))

