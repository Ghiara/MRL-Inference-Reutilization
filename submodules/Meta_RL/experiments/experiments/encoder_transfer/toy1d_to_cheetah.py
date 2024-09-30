import copy
import pprint
from configs.environment_factory import cheetah_goal, toy1d, toy1d_rand
from experiments.config_modules.encoder_decoration import *
from experiments.config_modules.inference import mlp_decoder, mdp_vae
from specific.encoder_transfer import map_cheetah_to_toy1d
from smrl.utility.ops import deep_dictionary_update

from smrl.policies.exploration import RandomPolicy

from smrl.policies.meta_policy import MetaRLTanhGaussianPolicy, PretrainedCheetah
from smrl.policies.meta_value_function import MlpValueFunction
from smrl.data_management.replay_buffers import TrajectoryReplayBuffer, MultiTaskReplayBuffer


# # Config which specifies RL training parameters (without encoder!)
no_transfer = dict(
    observation_dim = 1,
    action_dim = 1,

    # Policy networks
    qf_network_type = MlpValueFunction,
    qf_network_kwargs = {
        'hidden_sizes': [16,16,16],
    },
    policy_type = MetaRLTanhGaussianPolicy,
    policy_kwargs = {
        'hidden_sizes': [16,16,16],
    },

    # Replay buffers
    path_collector_kwargs = {},
    replay_buffer_type = TrajectoryReplayBuffer,
    replay_buffer_kwargs = {
        'max_path_number': 1_000,
        'max_sub_size': 1000,
    },
    inference_replay_buffer_type = MultiTaskReplayBuffer,
    inference_replay_buffer_kwargs = {
        'max_replay_buffer_size': 0,
        'max_sub_size': 0,
    },

    # Policy trainer
    policy_trainer_kwargs = dict(
        discount=0.99,
        policy_lr=1e-4,
        qf_lr=1e-4,
        encoder_lr=0,
        use_automatic_entropy_tuning=True,
        soft_target_tau=1e-3,
    ),

    # Algorithm
    algorithm_kwargs = dict(
        batch_size=128,
        num_epochs=5_000,
        max_path_length=250,
        prediction_target_size=0,        
        num_eval_paths_per_epoch=5,
        num_expl_paths_per_epoch=5,
        num_inference_paths_per_epoch=0,
        num_inference_trains_per_train_loop=0,
        num_policy_trains_per_train_loop=40,
        num_train_loops_per_epoch=25,
    )
)

no_transfer_config = dict(
    description = {
        "name": "TE-log"
    },

    latent_dim = 1,
    encoding_dim = 2,
    context_size = 2,

    environment_factory = toy1d,
    **no_transfer,

    **load_config_from_file("/home/ubuntu/juan/Meta-RL/data/delete/test_2023-12-18_15-31-19"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="/home/ubuntu/juan/Meta-RL/data/delete/test_2023-12-18_15-31-19",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        # io_modified_encoder(
        #     input_map=map_cheetah_to_toy1d,
        # ),
        encoding_mode='mean_var',
    ),
    **mlp_decoder([8,8,]),
    **mdp_vae(),
)
deep_dictionary_update(
    no_transfer_config,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'action_dim': 1,
        }
    )
)




# Config which specifies RL training parameters (without encoder!)
cheetah_training_config = dict(
    observation_dim = 20,
    action_dim = 6,

    # Policy networks
    qf_network_type = MlpValueFunction,
    qf_network_kwargs = {
        'hidden_sizes': [256,256,256],
    },
    policy_type = PretrainedCheetah,
    # policy_kwargs = {
    #     'hidden_sizes': [256,256,256],
    # },

    # Replay buffers
    path_collector_kwargs = {},
    replay_buffer_type = TrajectoryReplayBuffer,
    replay_buffer_kwargs = {
        'max_path_number': 1_000,
        'max_sub_size': 1000,
    },
    inference_replay_buffer_type = MultiTaskReplayBuffer,
    inference_replay_buffer_kwargs = {
        'max_replay_buffer_size': 0,
        'max_sub_size': 0,
    },

    # Policy trainer
    policy_trainer_kwargs = dict(
        discount=0.99,
        policy_lr=1e-4,
        qf_lr=1e-4,
        encoder_lr=0,
        use_automatic_entropy_tuning=True,
        soft_target_tau=1e-3,
    ),

    # Algorithm
    algorithm_kwargs = dict(
        batch_size=128,
        num_epochs=5_000,
        max_path_length=1_000,
        prediction_target_size=0,        
        num_eval_paths_per_epoch=5,
        num_expl_paths_per_epoch=5,
        num_inference_paths_per_epoch=0,
        num_inference_trains_per_train_loop=0,
        num_policy_trains_per_train_loop=40,
        num_train_loops_per_epoch=25,
    )
)

transfer_config = dict(
    description = {
        "name": "TE-log"
    },

    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    environment_factory = cheetah_goal,
    **cheetah_training_config,

    **load_config_from_file("/home/ubuntu/juan/Meta-RL/data/Baseline_increased_S_3_2023-12-09_11-32-46"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="/home/ubuntu/juan/Meta-RL/data/Baseline_increased_S_3_2023-12-09_11-32-46",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_cheetah_to_toy1d,
        ),
        encoding_mode='mean_var',
    ),
    **mlp_decoder([8,8,]),
    **mdp_vae(),
)
deep_dictionary_update(
    transfer_config,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'action_dim': 1,
        }
    )
)

transfer_config_strided = dict(
    description = {
        "name": "TE-log-strided(500-100)_uniform"
    },

    latent_dim = 1,
    encoding_dim = 2,
    context_size = 500,

    environment_factory = cheetah_goal,
    **cheetah_training_config,

    **load_config_from_file("/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_50_2023-12-07_18-40-02"),
    **multi_decorator(
        strided_encoder(stride=100),
        pretrained_encoder(
            path_to_data="/home/ubuntu/juan/Meta-RL/data/Baseline_max_task_50_2023-12-07_18-40-02",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_cheetah_to_toy1d,
        ),
        encoding_mode='mean_var',
    ),
    **mlp_decoder([8,8,]),
    **mdp_vae(),
)
deep_dictionary_update(
    transfer_config_strided,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'action_dim': 1 ,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)


if __name__ == "__main__":
    from experiments.experiments.runner import run_experiment

    configs = [
        # no_transfer_config
        transfer_config,
        # transfer_config_strided,
    ]

    for config in configs:
        # pprint.pprint(config, indent=2)
        run_experiment(
            config,
            gpu="0",
            multithreading=False,
            log_dir='./data/transfer/cheetah/presentation/'
        )