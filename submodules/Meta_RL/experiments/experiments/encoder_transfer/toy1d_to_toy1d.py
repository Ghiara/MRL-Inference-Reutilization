from configs.environment_factory import toy1d
from specific.environment_factory import *
from experiments.config_modules.encoder_decoration import *
from experiments.config_modules.inference import mlp_decoder, neural_process
from experiments.config_modules.policy import policy_networks
from experiments.config_modules.algorithm import only_policy_training, training_config

from smrl.utility.ops import deep_dictionary_update


config = dict(
    description = {
        "name": "TE-toy1d"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    environment_factory = toy1d,

    **pretrained_encoder(
        path_to_data="/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer/simple_exploration_action1/_2024-01-23_11-04-04",
        state_dict_keyword='trainer/Inference trainer/encoder',
        trainable=False,
        encoding_mode = 'mean_var',
    ),
    **mlp_decoder([8,8,]),
    **neural_process(),

    **policy_networks(),
    **only_policy_training(num_epochs=250, policy_lr=3e-4, qf_lr=3e-4),
)
deep_dictionary_update(
    config,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)

smaller_config = dict(
    description = {
        "name": "TE-toy1d"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    environment_factory = toy1d,

    **load_config_from_file("/home/ubuntu/juan/data/pretrained_for_transfer/exploration_only_policy/one_direction/_2024-01-31_08-29-43/"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="/home/ubuntu/juan/data/pretrained_for_transfer/exploration_only_policy/one_direction/_2024-01-31_08-29-43/",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        # io_modified_encoder(
        #     input_map=input_map_smaller,
        # ),
        encoding_mode = 'mean_var',
    ),
    **mlp_decoder([8,8,]),
    **neural_process(),

    **policy_networks(),
    **only_policy_training(num_epochs=1000, policy_lr=3e-4, qf_lr=3e-4),
)
deep_dictionary_update(
    smaller_config,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)

larger_config = dict(
    description = {
        "name": "TE-toy1d"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    environment_factory = toy1d_rand_larger,

    **load_config_from_file("/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=input_map_larger,
        ),
        encoding_mode = 'mean_var',
    ),
    **mlp_decoder([8,8,]),
    **neural_process(),

    **policy_networks(),
    **only_policy_training(num_epochs=250, policy_lr=3e-4, qf_lr=3e-4),
)
deep_dictionary_update(
    larger_config,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)

small_config = dict(
    description = {
        "name": "TE-toy1d"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    environment_factory = toy1d_rand_small,

    **load_config_from_file("/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=input_map_small,
        ),
        encoding_mode = 'mean_var',
    ),
    **mlp_decoder([8,8,]),
    **neural_process(),

    **policy_networks(),
    **only_policy_training(num_epochs=250, policy_lr=3e-4, qf_lr=3e-4),
)
deep_dictionary_update(
    small_config,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)

large_config = dict(
    description = {
        "name": "TE-toy1d"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    environment_factory = toy1d_rand_large,

    **load_config_from_file("/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=input_map_large,
        ),
        encoding_mode = 'mean_var',
    ),
    **mlp_decoder([8,8,]),
    **neural_process(),

    **policy_networks(),
    **only_policy_training(num_epochs=250, policy_lr=3e-4, qf_lr=3e-4),
)
deep_dictionary_update(
    large_config,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)

huge_config = dict(
    description = {
        "name": "TE-toy1d"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    environment_factory = toy1d_rand_huge,

    **load_config_from_file("/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=input_map_huge,
        ),
        encoding_mode = 'mean_var',
    ),
    **mlp_decoder([8,8,]),
    **neural_process(),

    **policy_networks(hidden_sizes=(64,64,64,64)),
    **only_policy_training(num_epochs=250, policy_lr=3e-4, qf_lr=3e-4),
)
deep_dictionary_update(
    huge_config,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)

huge_config_stride = dict(
    description = {
        "name": "TE-toy1d-strided"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 25,

    environment_factory = toy1d_rand_huge,

    **load_config_from_file("/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43"),
    **multi_decorator(
        strided_encoder(stride=5),
        pretrained_encoder(
            path_to_data="/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=input_map_huge,
        ),
        encoding_mode='mean_var',
    ),
    **mlp_decoder([8,8,]),
    **neural_process(),

    **policy_networks(hidden_sizes=(64,64,64,64)),
    **only_policy_training(num_epochs=250, policy_lr=3e-4, qf_lr=3e-4),
)
deep_dictionary_update(
    huge_config_stride,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)

huge_config_log = dict(
    description = {
        "name": "TE-toy1d-log"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    environment_factory = toy1d_rand_huge,

    **load_config_from_file("/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=input_map_huge,
        ),
        encoding_mode='mean_var',
    ),
    **mlp_decoder([8,8,]),
    **neural_process(),

    **policy_networks(hidden_sizes=(64,64,64,64)),
    **only_policy_training(num_epochs=250, policy_lr=3e-4, qf_lr=3e-4),
)
deep_dictionary_update(
    huge_config_log,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)

huge_config_log_stride = dict(
    description = {
        "name": "TE-toy1d-log-strided"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 25,

    environment_factory = toy1d_rand_huge,

    **load_config_from_file("/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43"),
    **multi_decorator(
        strided_encoder(stride=5),
        pretrained_encoder(
            path_to_data="/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=input_map_huge,
        ),
        encoding_mode = 'mean_var',
    ),
    **mlp_decoder([8,8,]),
    **neural_process(),

    **policy_networks(hidden_sizes=(64,64,64,64)),
    **only_policy_training(num_epochs=250, policy_lr=3e-4, qf_lr=3e-4),
)
deep_dictionary_update(
    huge_config_log_stride,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)

huge_config_attention = dict(
    description = {
        "name": "TE-toy1d-attention-np"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    environment_factory = toy1d_rand_huge,

    **load_config_from_file("/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="/home/ubuntu/juan/Meta-RL/data/old/Baseline_2023-11-18_19-42-43",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=input_map_huge,
        ),
        encoding_mode='mean_var',
    ),
    **mlp_decoder([8,8,]),
    **neural_process(),

    **policy_networks(hidden_sizes=(64,64,64,64)),
    **only_policy_training(num_epochs=250, policy_lr=3e-4, qf_lr=3e-4),
)
deep_dictionary_update(
    huge_config_log,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)


if __name__ == "__main__":
    from experiments.experiments.runner import run_experiment

    configs = [
        # config,
        smaller_config,
        # larger_config,
        # small_config,
        # large_config,
        # huge_config,
        # huge_config_log,
        # huge_config_stride,
        # huge_config_log_stride,
        # huge_config_attention,
    ]

    for config in configs:
        print(config)
        run_experiment(
            config,
            multithreading=True,
            log_dir='/home/ubuntu/juan/Meta-RL/data/pretrained_for_transfer'
        )