from configs.environment_factory import toy1d_without_boundary
from experiments.config_modules.inference import *
from experiments.config_modules.policy import policy_networks
from experiments.config_modules.algorithm import training_config
from experiments.config_modules.exploration import *


vae_config = dict(
    description = {
        "name": "vae"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    environment_factory = toy1d_without_boundary,

    **mlp_encoder(),
    **mlp_decoder(std_rew=1.5, std_obs=1.5),
    **mdp_vae(),
    **policy_networks(),
    **training_config(num_epochs=25_000,
                      inference_lr=1e-5),
    **log_random_memory_policy(
        action_update_interval = 10, 
        std_low = 0.001,
        std_high = 1.0,
        inference_only=True,
    ),
)

iwae_config = dict(
    description = {
        "name": "iwae"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    **gru_encoder(),
    **mlp_decoder(std_obs=5.0, std_rew=5.0),
    **mdp_iwae(),
    **policy_networks(),
    **training_config(),
    **multi_random_memory_policy(
        action_update_interval = 10, 
        mean_std_range = (0.01,1.0), 
        std_mean_range = (0.01,1.0), 
        S = 0.1, 
        inference_only=True,
    ),
)

np_config = dict(
    description = {
        "name": "np"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    **gru_encoder(),
    **mlp_decoder(std_obs=5.0, std_rew=5.0),
    **neural_process(),
    **policy_networks(),
    **training_config(),
    **multi_random_memory_policy(
        action_update_interval = 10, 
        mean_std_range = (0.01,1.0), 
        std_mean_range = (0.01,1.0), 
        S = 0.1, 
        inference_only=True,
    ),
)

iwnp_config = dict(
    description = {
        "name": "iwnp"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    **gru_encoder(),
    **mlp_decoder(std_obs=5.0, std_rew=5.0),
    **iw_neural_process(),
    **policy_networks(),
    **training_config(),
    **multi_random_memory_policy(
        action_update_interval = 10, 
        mean_std_range = (0.01,1.0), 
        std_mean_range = (0.01,1.0), 
        S = 0.1, 
        inference_only=True,
    ),
)


if __name__ == "__main__":
    from experiments.experiments.runner import run_experiment

    configs = [
        vae_config,
        iwae_config,
        np_config,
        iwnp_config,
    ]

    for config in configs:
        config['environment_factory'] = toy1d_without_boundary
        run_experiment(
            config,
            gpu="5",
            multithreading=True,
            log_dir='/data/bing/julius/experiments/toy1d_unlimited'
        )
