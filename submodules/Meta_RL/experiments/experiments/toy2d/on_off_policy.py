from configs.environment_factory import toy2d_rand
from experiments.config_modules.inference import mlp_decoder, neural_process, gru_encoder
from experiments.config_modules.policy import policy_networks
from experiments.config_modules.algorithm import *
from experiments.config_modules.exploration import *


on_policy = dict(
    description = {
        "name": "on-policy"
    },

    observation_dim = 2,
    action_dim = 2,
    latent_dim = 2,
    encoding_dim = 4,
    context_size = 5,

    **gru_encoder(encoding_mode='mean_var'),
    **mlp_decoder(),
    **neural_process(),
    **policy_networks(hidden_sizes=(32,32,32,32)),
    **training_config(num_epochs=10_000),
)

on_policy_entropy = dict(
    description = {
        "name": "on-policy-entropy-tuning"
    },

    observation_dim = 2,
    action_dim = 2,
    latent_dim = 2,
    encoding_dim = 4,
    context_size = 5,

    **gru_encoder(encoding_mode='mean_var'),
    **mlp_decoder(),
    **neural_process(),
    **policy_networks(hidden_sizes=(32,32,32,32)),
    **replay_buffers(),
    **inference_trainer_config(),
    **policy_trainer_config(use_automatic_entropy_tuning=True),
    **algorithm_config(num_epochs=10_000),
)

off_policy_inference_random = dict(
    description = {
        "name": "off-policy_random-inference"
    },

    observation_dim = 2,
    action_dim = 2,
    latent_dim = 2,
    encoding_dim = 4,
    context_size = 5,

    **gru_encoder(encoding_mode='mean_var'),
    **mlp_decoder(),
    **neural_process(),
    **policy_networks(hidden_sizes=(32,32,32,32)),
    **random_policy(std=0.05, inference_only=True),
    **training_config(num_epochs=10_000),
)

off_policy_inference_memory_random = dict(
    description = {
        "name": "off-policy_memory-random-inference"
    },

    observation_dim = 2,
    action_dim = 2,
    latent_dim = 2,
    encoding_dim = 4,
    context_size = 5,

    **gru_encoder(encoding_mode='mean_var'),
    **mlp_decoder(),
    **neural_process(),
    **policy_networks(hidden_sizes=(32,32,32,32)),
    **random_memory_policy(action_update_interval = 5, mean_std = 0.05, std_mean = 0.025, inference_only=True),
    **training_config(num_epochs=10_000),
)

off_policy_inference_multi_random = dict(
    description = {
        "name": "off-policy_multi-random-inference"
    },

    observation_dim = 2,
    action_dim = 2,
    latent_dim = 2,
    encoding_dim = 4,
    context_size = 5,

    **gru_encoder(encoding_mode='mean_var'),
    **mlp_decoder(),
    **neural_process(),
    **policy_networks(hidden_sizes=(32,32,32,32)),
    **multi_random_memory_policy(action_update_interval = 5, mean_std_range = (1e-5, 1e-1), std_mean_range = (1e-5, 1e-1), S = 1e-5, inference_only=True),
    **training_config(num_epochs=10_000),
)


off_policy_inference_log_random = dict(
    description = {
        "name": "off-policy_log-random-inference"
    },

    observation_dim = 2,
    action_dim = 2,
    latent_dim = 2,
    encoding_dim = 4,
    context_size = 5,

    **gru_encoder(encoding_mode='mean_var'),
    **mlp_decoder(),
    **neural_process(),
    **policy_networks(hidden_sizes=(32,32,32,32)),
    **log_random_memory_policy(action_update_interval = 5, std_low=1e-5, std_high=1e-1, inference_only=True),
    **training_config(num_epochs=10_000),
)

if __name__ == "__main__":
    from experiments.experiments.runner import run_experiment

    configs = [
        on_policy,
        on_policy_entropy,
        off_policy_inference_random,
        off_policy_inference_memory_random,
        off_policy_inference_multi_random,
        off_policy_inference_log_random,
    ]

    for config in configs:
        config['environment_factory'] = toy2d_rand
        run_experiment(
            config,
            gpu="0",
            multithreading=False,
            log_dir='/data/bing/julius/test/experiments/toy2d_on-off-policy'
        )
