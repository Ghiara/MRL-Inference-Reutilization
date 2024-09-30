import copy
from smrl.utility.ops import deep_dictionary_update
from smrl.vae.encoder_decorators.multi_decorator import MultiDecorator
from smrl.vae.encoder_decorators.striding import StridedEncoder
from specific.encoder_decorators import Toy1dForToy1dContinuousDecorator
from smrl.vae.encoder_networks import GRUEncoder, AttentionEncoder
from smrl.vae.mdpvae import MdpVAE
from smrl.policies.exploration import MultiRandomMemoryPolicy, RandomMemoryPolicy
from configs.base_configuration import config as config_
from smrl.vae.encoder_decorators.io_modification import InputMaskedEncoder
from smrl.vae.encoder_decorators.pretrained_encoder import PretrainedEncoder
from smrl.policies.meta_policy import MetaRLTanhGaussianPolicy
from smrl.policies.meta_value_function import MlpValueFunction

# ============ ENCODER TRAINING CONFIGS ===============

toy1d_config = dict(
    description = {
        'name': 'TE-Off-policy-base-config',
        'file': __file__,
        'variant': 'Toy1d off policy',
    },

    inference_policy_type = MultiRandomMemoryPolicy,   
    inference_policy_kwargs = {
        'action_update_interval': 25,
        'M': 0.0,
        'S': 2e-4,
        'mean_std_range': (0.0, 2e-4),
        'std_mean_range': (0.0, 2e-4),
        'sample_update_interval': True,
    },  
    algorithm_kwargs = dict(
        max_path_length=125,
        num_eval_paths_per_epoch=0,
        num_expl_paths_per_epoch=0,
        num_inference_paths_per_epoch=5,
        num_inference_trains_per_train_loop=5,
        num_policy_trains_per_train_loop=0,
    )
)
toy1d_config = deep_dictionary_update(copy.deepcopy(config_), toy1d_config)


vae_config = dict(
    description = {
        'name': 'TE-VAE-inference',
        'file': __file__,
        'variant': 'VAE inference',
    },
    inference_network_type = MdpVAE,
)
vae_config = deep_dictionary_update(copy.deepcopy(toy1d_config), vae_config)


beta_config = dict(
    description = {
        'name': 'TE-beta-10',
        'file': __file__,
        'variant': 'NP with beta = 10.0',
    },
    inference_network_kwargs = {
        'beta': 10.0,
    },
)
beta_config = deep_dictionary_update(copy.deepcopy(toy1d_config), beta_config)


multi_policy_config_ = dict(
    description = {
        'name': 'TE-multi-policy',
        'file': __file__,
        'variant': 'MultiRandomMemoryPolicy',
    },
    inference_policy_type = MultiRandomMemoryPolicy,   
    inference_policy_kwargs = {
        'action_update_interval': 10,
        'S': 0.05,
    },  
)
multi_policy_config = copy.deepcopy(toy1d_config)
multi_policy_config.update(multi_policy_config_)


input_masked_config = dict(
    description = {
        'name': 'TE-Input-masking',
        'file': __file__,
        'variant': 'Input-masked encoder',
    },
    encoder_decorator_type = InputMaskedEncoder,
    encoder_decorator_kwargs = {
        'mask': ['actions', 'next_observations', 'terminals'],
    },
)
input_masked_config = deep_dictionary_update(copy.deepcopy(toy1d_config), input_masked_config)


attention_encoder_config_ = dict(
    description = {
        'name': 'TE-attention-encoder',
        'file': __file__,
        'variant': 'Attention encoder',
    },

    encoder_type = AttentionEncoder,
    encoder_kwargs = dict(
        n_queries = 10,
        num_heads = 4,
        self_attention_layers = 2,
        query_layers = 2,
    ),
    
)
attention_encoder_config = copy.deepcopy(toy1d_config)
attention_encoder_config.update(attention_encoder_config_)





# ============== TRANSFER CONFIGS ====================

policy_config = dict(
    observation_dim = 2,
    encoder_decorator_type = Toy1dForToy1dContinuousDecorator,
    encoder_decorator_kwargs = {
        'path_to_weights': None,    # Overwrite in derived configurations!
        'trainable': False,
    },
    encoder_kwargs = {
        'observation_dim': 1,
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
        policy_lr=1e-2,
        qf_lr=1e-2,
        # clipping=1e2,               
        encoder_lr=None,
        use_automatic_entropy_tuning=False,
    ),
    algorithm_kwargs = dict(
        num_epochs=5_000,
        max_path_length=250,
        num_eval_paths_per_epoch=10,
        num_expl_paths_per_epoch=5,
        num_inference_paths_per_epoch=0,
        num_inference_trains_per_train_loop=0,
        num_policy_trains_per_train_loop=10,
        num_train_loops_per_epoch=10,
    )
)


standard_policy_config_ = dict(
    encoder_decorator_kwargs = {
        # 'path_to_weights': "data/transfer_encoders/toy1d_rand_TE-Off-policy-base-config/params.pkl",
        # 'path_to_weights': "data/encoder_training_log-policy_GRU/params.pkl",
        # 'path_to_weights': "data/encoder_training_strong-networks/params.pkl",
        'path_to_weights': "data/encoder_training/Encoder-training-scheduled/params.pkl",
        'state_dict_keyword': 'encoder',
    },
    encoder_kwargs = {
        "hidden_size": 64,
        "num_layers": 4,
    },
)
standard_policy_config = deep_dictionary_update(copy.deepcopy(toy1d_config), policy_config)
deep_dictionary_update(standard_policy_config, standard_policy_config_)


meanvar_policy_config_ = dict(
    description = {
        'name': 'TE-meanvar-config',
        'file': __file__,
        'variant': '\'mean_var\' encoding',
    },
    encoding_dim = 2,
    encoder_kwargs = {
        'encoding_mode': "mean_var",
    },
    encoder_decorator_kwargs = {
        'encoding_mode': "mean_var",
    }
)
meanvar_policy_config = deep_dictionary_update(copy.deepcopy(standard_policy_config), meanvar_policy_config_)

meanvar_long_policy_config = deep_dictionary_update(
    copy.deepcopy(meanvar_policy_config), 
    {
        'description': {
            'name': 'TE-meanvar_long-config',
            'file': __file__,
            'variant': '\'mean_var\' encoding, context size = 25',
        },
        'context_size': 25
    }
)


continuous_encoder_policy_config = dict(
    description = {
        'name': 'TE-continuous-encoder',
        'file': __file__,
        'variant': 'Continuous encoder',
    },
    encoding_dim = 2,
    context_size = 25,
    encoder_kwargs = {
        'encoding_mode': "mean_var",
        'observation_dim': 2,
        'hidden_size': 32,
        'num_layers': 4,
    },
    encoder_decorator_type = PretrainedEncoder,
    encoder_decorator_kwargs = {
        'encoding_mode': "mean_var",
        'path_to_weights': 'data/transfer_encoders/toy1d_cont_rand_Continuous-config-offpolicy/params.pkl',
        'state_dict_keyword': 'trainer/Inference trainer/encoder',
    }
)
continuous_encoder_policy_config = deep_dictionary_update(copy.deepcopy(standard_policy_config), continuous_encoder_policy_config)


standard_policy_config_strided_ = dict(
    description = {
        'name': 'TE-strided-config',
        'file': __file__,
        'variant': 'Striding 25 -> 5',
    },
    context_size = 25,
    encoder_kwargs = {
        **toy1d_config['encoder_kwargs'],
        'observation_dim': 1,
        'context_size': 5,
    },
    encoder_decorator_type = MultiDecorator,
    encoder_decorator_kwargs = {
        'decorators_and_kwargs': [
            (
                StridedEncoder,
                {
                    'stride': 5,
                }
            ),
            (
                Toy1dForToy1dContinuousDecorator,
                {
                    'path_to_weights': "data/transfer_encoders/toy1d_rand_TE-Off-policy-base-config/params.pkl",
                    'trainable': False,
                }
            ),
        ],
    }
)
standard_policy_config_strided = deep_dictionary_update(copy.deepcopy(toy1d_config), policy_config)
standard_policy_config_strided.update(standard_policy_config_strided_)

meanvar_policy_config_strided_ = dict(
    description = {
        'name': 'TE-meanvar-strided-config',
        'file': __file__,
        'variant': 'Striding 25 -> 5 & \'mean_var\'',
    },
    context_size = 25,
    encoding_dim = 2,
    encoder_kwargs = {
        "hidden_size": 64,
        "num_layers": 4,
        'observation_dim': 1,
        'context_size': 5,
        'encoding_mode': "mean_var",
    },
    encoder_decorator_type = MultiDecorator,
    encoder_decorator_kwargs = {
        'encoding_mode': "mean_var",
        'decorators_and_kwargs': [
            (
                StridedEncoder,
                {
                    'stride': 5,
                }
            ),
            (
                Toy1dForToy1dContinuousDecorator,
                {
                    # 'path_to_weights': "data/transfer_encoders/toy1d_rand_TE-Off-policy-base-config/params.pkl",
                    # 'path_to_weights': "data/encoder_training_log-policy_GRU/params.pkl",
                    # 'path_to_weights': "data/encoder_training_strong-networks/params.pkl",
                    'path_to_weights': "data/encoder_training/Encoder-training-scheduled/params.pkl",
                    'state_dict_keyword': 'encoder',
                    'trainable': False,
                }
            ),
        ],
    }
)
meanvar_policy_config_strided = deep_dictionary_update(copy.deepcopy(toy1d_config), policy_config)
meanvar_policy_config_strided.update(meanvar_policy_config_strided_)


vae_policy_config_ = dict(
    encoder_decorator_kwargs = {
        'path_to_weights': "data/transfer_encoders/toy1d_rand_TE-VAE-inference/params.pkl",
    }
)
vae_policy_config = deep_dictionary_update(copy.deepcopy(vae_config), policy_config)
deep_dictionary_update(vae_policy_config, vae_policy_config_)

vae_meanvar_policy_config = deep_dictionary_update(copy.deepcopy(vae_policy_config), meanvar_policy_config_)
vae_meanvar_policy_config['description'] = {
    'name': 'TE-vae-meanvar-config',
    'file': __file__,
    'variant': 'VAE \'mean_var\'',
}

vae_meanvar_strided_policy_config_ = dict(
    description = {
        'name': 'TE-VAE-meanvar-strided-config',
        'file': __file__,
        'variant': 'Striding 25 -> 5 & VAE & \'mean_var\'',
    },
    context_size = 25,
    encoding_dim = 2,
    encoder_kwargs = {
        **toy1d_config['encoder_kwargs'],
        'observation_dim': 1,
        'context_size': 5,
        'encoding_mode': "mean_var",
    },
    encoder_decorator_type = MultiDecorator,
    encoder_decorator_kwargs = {
        'encoding_mode': "mean_var",
        'decorators_and_kwargs': [
            (
                StridedEncoder,
                {
                    'stride': 5,
                }
            ),
            (
                Toy1dForToy1dContinuousDecorator,
                {
                    'path_to_weights': "data/transfer_encoders/toy1d_rand_TE-VAE-inference/params.pkl",
                    'trainable': False,
                }
            ),
        ],
    }
)
vae_meanvar_strided_policy_config = deep_dictionary_update(copy.deepcopy(toy1d_config), policy_config)
vae_meanvar_strided_policy_config.update(vae_meanvar_strided_policy_config_)


beta_policy_config_ = dict(
    encoder_decorator_kwargs = {
        'path_to_weights': "data/transfer_encoders/toy1d_rand_TE-beta-10/params.pkl",
    }
)
beta_policy_config = deep_dictionary_update(copy.deepcopy(beta_config), policy_config)
deep_dictionary_update(beta_policy_config, beta_policy_config_)


multi_policy_policy_config_ = dict(
    encoder_decorator_kwargs = {
        'path_to_weights': "data/transfer_encoders/toy1d_rand_TE-multi-policy/params.pkl",
    }
)
multi_policy_policy_config = deep_dictionary_update(copy.deepcopy(multi_policy_config), policy_config)
deep_dictionary_update(multi_policy_policy_config, multi_policy_policy_config_)


input_masked_policy_config_ = dict(
    encoder_decorator_type = MultiDecorator,
    encoder_decorator_kwargs = {
        'decorators_and_kwargs': [
            (
                InputMaskedEncoder,
                {
                    'mask': ['actions', 'next_observations', 'terminals']
                }
            ),
            (
                Toy1dForToy1dContinuousDecorator,
                {
                    'path_to_weights': "data/transfer_encoders/toy1d_rand_TE-Input-masking/params.pkl",
                    'trainable': False,
                }
            )
        ],
    }
)
input_masked_policy_config = deep_dictionary_update(copy.deepcopy(input_masked_config), policy_config)
input_masked_policy_config.update(input_masked_policy_config_)


attention_policy_config_ = dict(
    encoder_decorator_kwargs = {
        'path_to_weights': "data/transfer_encoders/toy1d_rand_TE-attention-encoder/params.pkl",
    }
)
attention_policy_config = deep_dictionary_update(copy.deepcopy(attention_encoder_config), policy_config)
deep_dictionary_update(attention_policy_config, attention_policy_config_)


noise_policy_config_ = dict(
    description = {
        'name': 'TE-noisy_env',
        'file': __file__,
        'variant': 'Noisy environment for encoder training',
    },
    encoder_decorator_kwargs = {
        'path_to_weights': "data/transfer_encoders/toy1d_rand_noisy_TE-Off-policy-base-config/params.pkl",
    }
)
noise_policy_config = deep_dictionary_update(copy.deepcopy(toy1d_config), policy_config)
deep_dictionary_update(noise_policy_config, noise_policy_config_)








# ========== Config print =============

if __name__ == '__main__':
    from pprint import pprint
    configs = copy.copy(locals())
    for name, value in configs.items():
        if not name.startswith("__"):
            pprint(name + ":")
            pprint(value)