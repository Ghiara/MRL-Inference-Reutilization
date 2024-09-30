from configs.environment_factory import toy1d_rand
from experiments.config_modules.inference import mlp_decoder, neural_process, gru_encoder
from experiments.config_modules.policy import policy_networks
from experiments.config_modules.algorithm import training_config
from experiments.config_modules.exploration import *


base_config = dict(

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    **gru_encoder(
        hidden_size=64,
        num_layers=4,
    ),
    **mlp_decoder(),
    **neural_process(),
    **policy_networks(),
    **training_config(
        inference_lr=3e-3,
        num_epochs=5_000,
        prediction_target_size=2,
        num_eval_paths_per_epoch=5,
        num_expl_paths_per_epoch=5,
        num_inference_paths_per_epoch=5,
        num_inference_trains_per_train_loop=10,
        num_policy_trains_per_train_loop=2,
    ),
)

on_policy = dict(
    description = {
        "name": "on-policy"
    },

    **base_config,
)

off_policy_inference_random = dict(
    description = {
        "name": "off-policy_random-inference"
    },

    **base_config,
    **random_policy(std=0.05, inference_only=True),
)

off_policy_inference_memory_random = dict(
    description = {
        "name": "off-policy_memory-random-inference"
    },

    **base_config,
    **random_memory_policy(action_update_interval = 10, mean_std = 0.05, std_mean = 0.025, inference_only=True),
)

off_policy_inference_multi_random = dict(
    description = {
        "name": "off-policy_multi-random-inference"
    },

    **base_config,
    **multi_random_memory_policy(action_update_interval = 5, mean_std_range = (1e-5, 1e-1), std_mean_range = (1e-5, 1e-1), S = 1e-5, inference_only=True),
)

off_policy_inference_log_random = dict(
    description = {
        "name": "off-policy_log-random-inference"
    },

    **base_config,
    **log_random_memory_policy(action_update_interval = 5, std_low=1e-5, std_high=1e-1, inference_only=True),
)

if __name__ == "__main__":
    from experiments.experiments.runner import run_experiment

    configs = [
        on_policy,
        off_policy_inference_random,
        off_policy_inference_memory_random,
        off_policy_inference_log_random,
        off_policy_inference_multi_random,
    ]

    for config in configs:
        config['environment_factory'] = toy1d_rand
        run_experiment(
            config,
            gpu="",
            multithreading=True,
            log_dir='/data/bing/julius/experiments/toy1d_on-off-policy'
        )
