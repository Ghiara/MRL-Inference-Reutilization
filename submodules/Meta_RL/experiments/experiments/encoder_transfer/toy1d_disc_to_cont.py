import copy
import pprint
from configs.environment_factory import toy1d_cont_rand
from experiments.config_modules.encoder_decoration import *
from experiments.config_modules.inference import mlp_decoder, neural_process
from experiments.config_modules.policy import policy_networks
from experiments.config_modules.algorithm import only_policy_training
from specific.encoder_transfer import map_toy1d_cont_to_disc
from smrl.utility.ops import deep_dictionary_update


base_config = dict(
    observation_dim = 2,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,

    environment_factory = toy1d_cont_rand,
    **mlp_decoder([8,8,]),
    **neural_process(),

    **policy_networks(hidden_sizes=(32,32,32,32)),
    **only_policy_training(num_epochs=500, policy_lr=3e-4, qf_lr=3e-4),
)


# On-policy encoder
transfer_config_on_policy = dict(
    **base_config,

    description = {
        "name": "TE-toy1d-on-policy"
    },

    context_size = 5,

    **load_config_from_file("data/transfer_encoders/toy1d_on-policy"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="data/transfer_encoders/toy1d_on-policy",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_toy1d_cont_to_disc,
        ),
        encoding_mode='mean_var',
    ),
)
deep_dictionary_update(
    transfer_config_on_policy,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
        }
    )
)

transfer_config_on_policy_strided = dict(
    **base_config,
    description = {
        "name": "TE-toy1d-on-policy-strided"
    },

    context_size = 25,

    **load_config_from_file("data/transfer_encoders/toy1d_on-policy"),
    **multi_decorator(
        strided_encoder(stride=5),
        pretrained_encoder(
            path_to_data="data/transfer_encoders/toy1d_on-policy",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_toy1d_cont_to_disc,
        ),
        encoding_mode='mean_var',
    ),
)
deep_dictionary_update(
    transfer_config_on_policy_strided,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)

# Memory random policy encoder
transfer_config_rand = dict(
    **base_config,

    description = {
        "name": "TE-toy1d-rand"
    },

    context_size = 5,

    **load_config_from_file("data/transfer_encoders/toy1d_rand"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="data/transfer_encoders/toy1d_rand",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_toy1d_cont_to_disc,
        ),
        encoding_mode='mean_var',
    ),
)
deep_dictionary_update(
    transfer_config_rand,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
        }
    )
)

transfer_config_rand_strided = dict(
    **base_config,
    description = {
        "name": "TE-toy1d-rand-strided"
    },

    context_size = 25,

    **load_config_from_file("data/transfer_encoders/toy1d_rand"),
    **multi_decorator(
        strided_encoder(stride=5),
        pretrained_encoder(
            path_to_data="data/transfer_encoders/toy1d_rand",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_toy1d_cont_to_disc,
        ),
        encoding_mode='mean_var',
    ),
)
deep_dictionary_update(
    transfer_config_rand_strided,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)


# Memory random policy encoder
transfer_config_memory = dict(
    **base_config,

    description = {
        "name": "TE-toy1d-memory"
    },

    context_size = 5,

    **load_config_from_file("data/transfer_encoders/toy1d_memory-rand"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="data/transfer_encoders/toy1d_memory-rand",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_toy1d_cont_to_disc,
        ),
        encoding_mode='mean_var',
    ),
)
deep_dictionary_update(
    transfer_config_memory,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
        }
    )
)

transfer_config_memory_strided = dict(
    **base_config,
    description = {
        "name": "TE-toy1d-memory-strided"
    },

    context_size = 25,

    **load_config_from_file("data/transfer_encoders/toy1d_memory-rand"),
    **multi_decorator(
        strided_encoder(stride=5),
        pretrained_encoder(
            path_to_data="data/transfer_encoders/toy1d_memory-rand",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_toy1d_cont_to_disc,
        ),
        encoding_mode='mean_var',
    ),
)
deep_dictionary_update(
    transfer_config_memory_strided,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)


# Multi-random policy encoder
transfer_config_multi = copy.deepcopy(transfer_config_memory)
transfer_config_multi.update(dict(
    description = {
        "name": "TE-toy1d-multi"
    },

    **load_config_from_file("data/transfer_encoders/toy1d_multi-rand"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="data/transfer_encoders/toy1d_multi-rand",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_toy1d_cont_to_disc,
        ),
        encoding_mode='mean_var',
    ),
))
deep_dictionary_update(
    transfer_config_multi,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
        }
    )
)

transfer_config_multi_strided = dict(
    **base_config,
    description = {
        "name": "TE-toy1d-multi-strided"
    },

    context_size = 25,

    **load_config_from_file("data/transfer_encoders/toy1d_multi-rand"),
    **multi_decorator(
        strided_encoder(stride=5),
        pretrained_encoder(
            path_to_data="data/transfer_encoders/toy1d_multi-rand",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_toy1d_cont_to_disc,
        ),
        encoding_mode='mean_var',
    ),
)
deep_dictionary_update(
    transfer_config_multi_strided,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)


# Log-random policy encoder
transfer_config_log = copy.deepcopy(transfer_config_memory)
transfer_config_log.update(dict(
    description = {
        "name": "TE-toy1d-log"
    },

    **load_config_from_file("data/transfer_encoders/toy1d_log-rand"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="data/transfer_encoders/toy1d_log-rand",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_toy1d_cont_to_disc,
        ),
        encoding_mode='mean_var',
    ),
)),
deep_dictionary_update(
    transfer_config_log,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
        }
    )
)

transfer_config_log_strided = dict(
    **base_config,
    description = {
        "name": "TE-toy1d-log-strided"
    },
    context_size = 25,

    **load_config_from_file("data/transfer_encoders/toy1d_log-rand"),
    **multi_decorator(
        strided_encoder(stride=5),
        pretrained_encoder(
            path_to_data="data/transfer_encoders/toy1d_log-rand",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_toy1d_cont_to_disc,
        ),
        encoding_mode='mean_var',
    ),
)
deep_dictionary_update(
    transfer_config_log_strided,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)


attention_config = dict(
    **base_config,
    description = {
        "name": "TE-toy1d-attention"
    },
    context_size = 5,

    **load_config_from_file("data/transfer_encoders/toy1d_attention-np"),
    **multi_decorator(
        pretrained_encoder(
            path_to_data="data/transfer_encoders/toy1d_attention-np",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_toy1d_cont_to_disc,
        ),
        encoding_mode='mean_var',
    ),
)
deep_dictionary_update(
    attention_config,
    dict(
        encoder_kwargs = {
            'observation_dim': 1,
            'encoding_mode': 'mean_var',
            'context_size': 5,
        }
    )
)


attention_config_strided = dict(
    **base_config,
    description = {
        "name": "TE-toy1d-attention-strided"
    },
    context_size = 25,

    **load_config_from_file("data/transfer_encoders/toy1d_attention-np"),
    **multi_decorator(
        strided_encoder(stride=5),
        pretrained_encoder(
            path_to_data="data/transfer_encoders/toy1d_attention-np",
            state_dict_keyword='trainer/Inference trainer/encoder',
            load_config=False,
            trainable=False,
        ),
        io_modified_encoder(
            input_map=map_toy1d_cont_to_disc,
        ),
        encoding_mode='mean_var',
    ),
)
deep_dictionary_update(
    attention_config_strided,
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
        transfer_config_log,
        transfer_config_on_policy,
        transfer_config_rand,
        transfer_config_memory,
        transfer_config_multi,
        transfer_config_log_strided,
        transfer_config_on_policy_strided,
        transfer_config_rand_strided,
        transfer_config_memory_strided,
        transfer_config_multi_strided,
        attention_config,
        attention_config_strided,
    ]

    for config in configs:
        pprint.pprint(config, indent=2)
        run_experiment(
            config,
            gpu="4,6",
            multithreading=True,
            log_dir='/data/bing/julius/experiments/toy1d_TE_disc-to-cont_2'
        )