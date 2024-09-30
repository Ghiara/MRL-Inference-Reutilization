import torch
from smrl.vae.encoder_networks import GRUEncoder, MlpEncoder
from smrl.vae.decoder_networks import MlpDecoder, SeparateMlpDecoder
from smrl.vae.mdpvae import NeuralProcess, MdpVAE, InfoMaxMdpVAE
from smrl.policies.meta_policy import MetaRLTanhGaussianPolicy
from smrl.policies.meta_value_function import MlpValueFunction
from smrl.policies.exploration import RandomPolicy, RandomMemoryPolicy, MultiRandomMemoryPolicy
from smrl.data_management.replay_buffers import ContextReplayBuffer, TrajectoryReplayBuffer, MultiTaskReplayBuffer
from collections import OrderedDict

config = OrderedDict(

    description = {
        'name': 'Base-config',
        'file': __file__,
        'variant': 'Base configuration',
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 1,   # Can be different to latent_dim, e.g. if encoder.encoding_mode is equal to 'mean_var'
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
    qf_network_type = MlpValueFunction,
    qf_network_kwargs = {
        'hidden_sizes': [64, 64, 64, 64],
    },
    policy_type = MetaRLTanhGaussianPolicy,
    policy_kwargs = {
        'hidden_sizes': [64, 64, 64, 64],
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
    # expl_policy_type = MultiRandomMemoryPolicy,
    # expl_policy_kwargs = {
    #     'action_update_interval': 10,
    #     'std': 0.05,
    # },
    # inference_policy_type = MultiRandomMemoryPolicy,    # Exploration policy for off-policy inference training (None -> on-policy training)
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
        policy_lr=3e-4,             # Learning rate for the policy network
        qf_lr=3e-4,                 # Learning rate for the Q-function networks
        # clipping=1e2,               
        encoder_lr=None,
        use_automatic_entropy_tuning=False,
    ),
    algorithm_kwargs = dict(
        batch_size=128,             # Batch size for batch-training (= number of samples which is passed to the trainers in each training step)
        num_epochs=5_000,             # Number of training epochs
        max_path_length=250,        # Maximum path length
        prediction_target_size=5,  # Size of the prediction targets for ELBO computation (number of transitions which the decoder needs to predict (reward & next observation))
        num_eval_paths_per_epoch=15,    # Number of trajectories used for evaluation
        num_expl_paths_per_epoch=10,    # Number of trajectories added to the policy replay buffer in each train loop
        num_inference_paths_per_epoch=10,   # Number of trajectories added to the inference replay buffer in each train loop
        num_inference_trains_per_train_loop=5,  # Number of inference training steps in each train loop
        num_policy_trains_per_train_loop=10, # Number of policy training steps in each train loop
        num_train_loops_per_epoch=20,    # Number of train loops per epoch
    ),
)
