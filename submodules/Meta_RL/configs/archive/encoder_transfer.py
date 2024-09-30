import copy
from pathlib import Path

from smrl.utility.ops import deep_dictionary_update

from smrl.policies.meta_value_function import MlpValueFunction
from smrl.policies.meta_policy import MetaRLTanhGaussianPolicy
from smrl.vae.decoder_networks import SeparateMlpDecoder
from smrl.vae.mdpvae import NeuralProcess
from smrl.data_management.replay_buffers import TrajectoryReplayBuffer

from smrl.vae.encoder_networks import GRUEncoder
from smrl.vae.encoder_decorators import PretrainedEncoder
from smrl.vae.encoder_decorators.multi_decorator import MultiDecorator
from smrl.vae.encoder_decorators.striding import StridedEncoder

from specific.encoder_decorators import Toy1dForToy1dContinuousDecorator


"""
Path and Encoder --> TODO
"""
# path_to_data = "data/encoder_training/Encoder-training-scheduled"
# state_dict_keyword = 'encoder'
# encoder_type = GRUEncoder
# encoder_kwargs = {
#     'hidden_size': 64,
#     'num_layers': 4,
# }

# path_to_data = "data/transfer_encoders/toy1d_rand_TE-Off-policy-base-config"
# state_dict_keyword = 'trainer/Inference trainer/encoder'
# encoder_type = GRUEncoder
# encoder_kwargs = {
#     'hidden_size': 32,
#     'num_layers': 4,
# }

path_to_data = "data/transfer_encoders/toy1d_cont_rand_Continuous-config-offpolicy"
state_dict_keyword = 'trainer/Inference trainer/encoder'
encoder_type = GRUEncoder
encoder_kwargs = {
    'hidden_size': 32,
    'num_layers': 4,
}


path_to_data = Path(path_to_data)
path_to_weights = str(path_to_data.joinpath("params.pkl"))




# ============== TRANSFER CONFIGS ====================

"""
Policy training configuration
"""
policy_config = dict(
    description = None,

    action_dim = 1,
    observation_dim = None,
    latent_dim = None,
    encoding_dim = None,
    context_size = None,

    encoder_type = encoder_type,
    encoder_kwargs = encoder_kwargs,

    decoder_type = SeparateMlpDecoder,
    decoder_kwargs = {
        'hidden_sizes': [8, 8,],
        'std_rew': 0.1,
        'std_obs': 0.1,
    },
    inference_network_type = NeuralProcess,
    inference_network_kwargs = {
        'beta': 1.0,
    },

    qf_network_type = MlpValueFunction,
    qf_network_kwargs = {
        'hidden_sizes': [32, 32, 32, 32,],
    },
    policy_type = MetaRLTanhGaussianPolicy,
    policy_kwargs = {
        'hidden_sizes': [16,16,16],
    },
    policy_trainer_kwargs = dict(
        discount=0.99,
        policy_lr=1e-4,
        qf_lr=1e-4,
        # clipping=1e2,               
        encoder_lr=None,
        use_automatic_entropy_tuning=False,
    ),
    replay_buffer_type = TrajectoryReplayBuffer,
    replay_buffer_kwargs = {
        'max_path_number': 1_000,
        'max_sub_size': 300,
    },
    algorithm_kwargs = dict(
        num_epochs=2500,
        batch_size=128,
        max_path_length=250,
        num_eval_paths_per_epoch=10,
        num_expl_paths_per_epoch=5,
        num_inference_paths_per_epoch=0,    # !
        num_inference_trains_per_train_loop=0,  # !
        num_policy_trains_per_train_loop=10,
        num_train_loops_per_epoch=10,
    )
)


"""
Toy1D encoder > Toy1D policy
"""

discrete_discrete = deep_dictionary_update(
    copy.deepcopy(policy_config),
    dict(
        description = {
            'name': 'TE_' + path_to_data.name,
            'variant': "Discrete > Discrete",
        },

        observation_dim = 1,
        latent_dim = 1,
        encoding_dim = 1,
        context_size = 5,

        encoder_decorator_type = PretrainedEncoder,
        encoder_decorator_kwargs = {
            'path_to_weights': path_to_weights,
            'trainable': False,
            'state_dict_keyword': state_dict_keyword,
        },
    )
)

discrete_discrete_meanvar = deep_dictionary_update(
    copy.deepcopy(discrete_discrete),
    dict(
        description = {
            'name': 'TE_' + path_to_data.name + "_meanvar",
            'variant': "Discrete > Discrete, \'mean_var\'",
        },
        encoding_dim = 2,
        encoder_kwargs = {
            'encoding_mode': 'mean_var',
        },
        encoder_decorator_kwargs = {
            'encoding_mode': 'mean_var',
        },
    )
)


"""
Toy1D encoder > cont. Toy1D policy
"""

discrete_continuous = deep_dictionary_update(
    copy.deepcopy(policy_config),
    dict(
        description = {
            'name': 'TE_' + path_to_data.name,
            'variant': 'Discrete > Continuous',
        },

        observation_dim = 2,
        latent_dim = 1,
        encoding_dim = 1,
        context_size = 5,

        encoder_kwargs = {
            'observation_dim': 1,
        },

        encoder_decorator_type = Toy1dForToy1dContinuousDecorator,
        encoder_decorator_kwargs = {
            'path_to_weights': path_to_weights,
            'trainable': False,
            'state_dict_keyword': state_dict_keyword,
        },
    )
)

discrete_continuous_meanvar = deep_dictionary_update(
    copy.deepcopy(discrete_continuous),
    dict(
        description = {
            'name': 'TE_' + path_to_data.name + "_meanvar",
            'variant': 'Discrete > Continuous, \'mean_var\''
        },
        encoding_dim = 2,
        encoder_kwargs = {
            'encoding_mode': 'mean_var',
        },
        encoder_decorator_kwargs = {
            'encoding_mode': 'mean_var',
        },
    )
)

discrete_continuous_strided = deep_dictionary_update(
    copy.deepcopy(policy_config),
    dict(
        description = {
            'name': 'TE_' + path_to_data.name + "_strided",
            'variant': 'Discrete > Continuous, strided & \'mean_var\''
        },

        observation_dim = 2,
        latent_dim = 1,
        encoding_dim = 2,
        context_size = 25,

        encoder_kwargs = {
            'observation_dim': 1,
            'context_size': 5,
            'encoding_mode': 'mean_var',
        },

        encoder_decorator_type = MultiDecorator,
        encoder_decorator_kwargs = dict(
            encoding_mode = 'mean_var',
            decorators_and_kwargs = [
                (
                    StridedEncoder, 
                    {'stride': 5}
                ),
                (
                    Toy1dForToy1dContinuousDecorator,
                    {
                        'path_to_weights': path_to_weights,
                        'trainable': False,
                        'state_dict_keyword': state_dict_keyword
                    }
                )
            ]
        ),
    )
)


"""
cont. Toy1D encoder > cont. Toy1D policy
"""
continuous_continuous = deep_dictionary_update(
    copy.deepcopy(policy_config),
    dict(
        description = {
            'name': 'TE_' + path_to_data.name,
            'variant': 'Continuous > Continuous',
        },

        observation_dim = 2,
        latent_dim = 1,
        encoding_dim = 1,
        context_size = 5,

        encoder_decorator_type = PretrainedEncoder,
        encoder_decorator_kwargs = {
            'path_to_weights': path_to_weights,
            'trainable': False,
            'state_dict_keyword': state_dict_keyword,
        },
    )
)

continuous_continuous_meanvar = deep_dictionary_update(
    copy.deepcopy(continuous_continuous),
    dict(
        description = {
            'name': 'TE_' + path_to_data.name + "_meanvar",
            'variant': 'Discrete > Continuous, \'mean_var\''
        },
        encoding_dim = 2,
        encoder_kwargs = {
            'encoding_mode': 'mean_var',
        },
        encoder_decorator_kwargs = {
            'encoding_mode': 'mean_var',
        },
    )
)
