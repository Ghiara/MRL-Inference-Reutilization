import copy
from smrl.utility.ops import deep_dictionary_update

from configs.environment_factory import toy1d_rand
from experiments.config_modules.inference import mlp_decoder, neural_process, gru_encoder
from experiments.config_modules.policy import policy_networks
from experiments.config_modules.algorithm import inference_trainer_config, policy_trainer_config, algorithm_config
from experiments.config_modules.exploration import *

from smrl.data_management.replay_buffers import *
import os

# Inference buffer: ContextReplayBuffer
context_buffer = dict(
    description = {
        "name": "context-buffer"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 1,
    context_size = 5,

    **gru_encoder(),
    **mlp_decoder(),
    **neural_process(),
    **policy_networks(),
    **random_memory_policy(action_update_interval = 10, mean_std = 0.05, std_mean = 0.025, inference_only=True),
    **inference_trainer_config(),
    **policy_trainer_config(),
    **algorithm_config(),

    path_collector_kwargs = {},
    replay_buffer_type = TrajectoryReplayBuffer,
    replay_buffer_kwargs = {
        'max_path_number': 1_000,
        'max_sub_size': 300,
    },

    inference_replay_buffer_type = ContextReplayBuffer,
    inference_replay_buffer_kwargs = {
        'max_replay_buffer_size': 50_000,
        'randomize_contexts': False,
        'randomize_targets': False
    }
)

# Inference buffer: TrajectoryReplayBuffer
trajectory_buffer_ordered = copy.deepcopy(context_buffer)
trajectory_buffer_ordered.update(dict(
    description = {
        "name": "trajectory-buffer-ordered"
    },
    inference_replay_buffer_type = TrajectoryReplayBuffer,
    inference_replay_buffer_kwargs = {
        'max_path_number': 1_000,
        'max_sub_size': 300,
        'randomize_targets': False,
        'randomize_contexts': False,
    },
))

trajectory_buffer_randomized = deep_dictionary_update(
    copy.deepcopy(trajectory_buffer_ordered),
    dict(
        description = {
            "name": "trajectory-buffer-randomized"
        },
        inference_replay_buffer_kwargs = {
            'randomize_targets': True,
        },
    )
)

# Inference buffer: MultiTaskReplayBuffer
multitask_buffer_ordered = copy.deepcopy(context_buffer)
multitask_buffer_ordered.update(dict(
    description = {
        "name": "multitask-buffer-ordered"
    },
    inference_replay_buffer_type = MultiTaskReplayBuffer,
    inference_replay_buffer_kwargs = {
        'max_replay_buffer_size': 50_000,
        'max_sub_size': 2_500,
        'randomize_targets': False,
        'randomize_contexts': False,
    }
))

multitask_buffer_randomized = deep_dictionary_update(
    copy.deepcopy(multitask_buffer_ordered),
    dict(
        description = {
            "name": "multitask-buffer-randomized"
        },
        inference_replay_buffer_kwargs = {
            'randomize_targets': True,
        },
    )
)



if __name__ == "__main__":
    from experiments.experiments.runner import run_experiment

    configs = [
        context_buffer,
        trajectory_buffer_ordered,
        trajectory_buffer_randomized,
        multitask_buffer_ordered,
        multitask_buffer_randomized,
    ]

    for config in configs:
        config['environment_factory'] = toy1d_rand
        run_experiment(
            config,
            multithreading=True,
            log_dir=f'{os.getcwd()}/data/delete'
        )
