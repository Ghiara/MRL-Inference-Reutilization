import os
import copy

from smrl.trainers.vae import MdpVAETrainer
from smrl.data_management.vae_training_data import ContextTargetTaskBuffer, ContextCollector
from smrl.vae.encoder_networks import GRUEncoder, AttentionEncoder, MlpEncoder
from smrl.vae.decoder_networks import SeparateMlpDecoder
from smrl.vae.mdpvae import NeuralProcess, IWNeuralProcess, MdpVAE, MdpIWAE
from smrl.policies.exploration import LogMultiRandomMemoryPolicy, RandomMemoryPolicy, MultiRandomMemoryPolicy
from specific.env_reset_functions import toy1d_borderless_reset

from smrl.experiment.encoder_training import init_algorithm
from smrl.utility.ops import deep_dictionary_update

from configs.environment_factory import *

from experiments.config_modules.beta_schedules import *

    
small_std = dict(
    description = {
        'name': 'small-std',
    },

    environment_factory = toy1d_without_boundary,

    inference_policy_type = RandomMemoryPolicy,
    # inference_policy_kwargs = dict(
    #     action_update_interval=25,
    #     std_low=1e-3, 
    #     std_high=5e0,
    #     mean_mean=0.0, 
    #     sample_update_interval=False
    # ),
    inference_policy_kwargs = dict(
        action_update_interval = 5,
        std_mean = 1.0,
        mean_std = 1.0,
        sample_update_interval = False,
    ),

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    # Inference mechanism
    encoder_type = GRUEncoder, 
    encoder_kwargs = {
        'hidden_size': 32,
        'num_layers': 4,
        'encoding_mode': 'mean_var',
    },
    decoder_type = SeparateMlpDecoder, 
    decoder_kwargs = {
        'hidden_sizes': [16, 16, ],
        'std_rew': 0.5,
        'std_obs': 0.1,
        'train_std': False,
    }, 
    inference_network_type = NeuralProcess, 
    inference_network_kwargs = {
        'beta': 1.0,
    },

    collector_type = ContextCollector,
    collector_kwargs = {
        'target_size': 5,
        'reset_function': toy1d_borderless_reset,
    },

    buffer_type = ContextTargetTaskBuffer,
    buffer_kwargs = {
        'maxlen': 20
    },

    trainer_type = MdpVAETrainer,
    trainer_kwargs = {
        'lr': 3e-4,
        'n_latent_samples': 128,
    },

    algorithm_kwargs = {
        'n_epochs': 25_000,
        'batch_size': 128,
        'train_calls_per_epoch': 50,
        'samples_per_epoch': 20,
        'initial_samples': 5000,
    },

)

large_std = deep_dictionary_update(
    copy.deepcopy(small_std),
    dict(
        description = {
            'name': 'large-std',
        },
        decoder_kwargs = {
            'std_rew': 5.0,
            'std_obs': 1.0,
        }, 
    )
)

very_large_std = deep_dictionary_update(
    copy.deepcopy(small_std),
    dict(
        description = {
            'name': 'very-large-std',
        },
        decoder_kwargs = {
            'std_rew': 10.0,
            'std_obs': 10.0,
        }, 
    )
)

trainable_std = deep_dictionary_update(
    copy.deepcopy(very_large_std),
    dict(
        description = {
            'name': 'trainable-std',
        },
        decoder_kwargs = {
            'train_std': True,
        }, 
    )
)

very_large_std_batchnorm = deep_dictionary_update(
    copy.deepcopy(very_large_std),
    dict(
        description = {
            'name': 'very-large-std-batchnorm'
        },
        encoder_kwargs = {
            'batch_norm': True
        }
    )
)

trainable_std_batchnorm = deep_dictionary_update(
    copy.deepcopy(trainable_std),
    dict(
        description = {
            'name': 'trainable-std-batchnorm'
        },
        encoder_kwargs = {
            'batch_norm': True
        }
    )
)

importance_weighted_large = deep_dictionary_update(
    copy.deepcopy(very_large_std),
    dict(
        description = {
            'name': 'importance-weighted-very-large-std',
        },
        inference_network_type = IWNeuralProcess, 
    )
)

importance_weighted_trainable = deep_dictionary_update(
    copy.deepcopy(trainable_std),
    dict(
        description = {
            'name': 'importance-weighted-trainable-std',
        },
        inference_network_type = IWNeuralProcess, 
    )
)

scheduled = deep_dictionary_update(
    copy.deepcopy(trainable_std),
    dict(
        description = {
            'name': 'scheduled-beta',
        },
        trainer_kwargs = dict(
            beta_schedule = cyclic_schedule
        )
    )
)

scheduled_regularized = deep_dictionary_update(
    copy.deepcopy(trainable_std),
    dict(
        description = {
            'name': 'scheduled-beta-reg',
        },
        trainer_kwargs = dict(
            beta_schedule = cyclic_schedule_regularization,
        )
    )
)

mdpvae_large_std = deep_dictionary_update(
    copy.deepcopy(large_std),
    dict(
        description = {
            'name': 'vae-large-std',
        },
        inference_network_type = MdpVAE, 
    )
)

mdpvae_very_large_std = deep_dictionary_update(
    copy.deepcopy(very_large_std),
    dict(
        description = {
            'name': 'vae-very-large-std',
        },
        inference_network_type = MdpVAE, 
    )
)

mdpvae_very_large_std_scheduled = deep_dictionary_update(
    copy.deepcopy(very_large_std),
    dict(
        description = {
            'name': 'vae-very-large-std-scheduled',
        },
        inference_network_type = MdpVAE, 
        trainer_kwargs = dict(
            beta_schedule = cyclic_schedule
        ),
    )
)

mdpvae_mlp_encoder = copy.deepcopy(mdpvae_very_large_std)
mdpvae_mlp_encoder.update(dict(
    description = {
            'name': 'vae-mlp-encoder',
        },
    encoder_type = MlpEncoder, 
    encoder_kwargs = {
        'hidden_sizes': (32,32,32,32),
        'encoding_mode': 'mean_var',
    },
))

mdpiwae = deep_dictionary_update(
    copy.deepcopy(very_large_std),
    dict(
        description = {
            'name': 'iwae',
        },
        inference_network_type = MdpIWAE, 
    )
)


if __name__ == '__main__':
    # Set GPU device as indexed by nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # Choose configs to run
    configs = [
        # small_std,
        # large_std,
        # very_large_std,
        # trainable_std,
        # trainable_std_batchnorm,
        # very_large_std_batchnorm,
        # importance_weighted_large,
        # importance_weighted_trainable,
        # scheduled,
        # scheduled_regularized,
        # mdpvae,
        # mdpvae_large_std,
        # mdpvae_very_large_std,
        # mdpvae_very_large_std_scheduled,
        mdpvae_mlp_encoder,
        # mdpiwae,
    ]

    for config in configs:
        # experiment_name = config["environment_factory"].__name__ + "_" + config['description']['name']
        experiment_name = config['description']['name']
        logger_kwargs = dict(
            log_dir=os.path.join("/data/bing/julius/encoder_training/toy1d_without_boundary", experiment_name),
            snapshot_mode='last',
            snapshot_gap=25,
        )

    algorithm = init_algorithm(config, logger_kwargs=logger_kwargs)
    algorithm.train()