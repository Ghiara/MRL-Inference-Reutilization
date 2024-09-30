import os
import copy

from smrl.trainers.vae import MdpVAETrainer
from smrl.data_management.vae_training_data import ContextTargetTaskBuffer, ContextCollector
from smrl.vae.encoder_networks import GRUEncoder, AttentionEncoder
from smrl.vae.decoder_networks import SeparateMlpDecoder
from smrl.vae.mdpvae import NeuralProcess, IWNeuralProcess
from smrl.policies.exploration import LogMultiRandomMemoryPolicy, RandomMemoryPolicy, MultiRandomMemoryPolicy

from smrl.experiment.encoder_training import init_algorithm
from smrl.utility.ops import deep_dictionary_update

from configs.environment_factory import *

from experiments.config_modules.beta_schedules import cyclic_schedule, cyclic_schedule_regularization


log_policy = dict(
    description = {
        'name': 'log-policy',
    },

    environment_factory = toy1d_rand,

    inference_policy_type = LogMultiRandomMemoryPolicy,
    inference_policy_kwargs = dict(
        action_update_interval=25,
        std_low=1e-4, 
        std_high=1e-1,
        mean_mean=0.0, 
        sample_update_interval=False
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
        'std_rew': 0.1,
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


trainable_std = deep_dictionary_update(
    copy.deepcopy(log_policy),
    dict(
        description = {
            'name': 'trainable-std',
        },
        decoder_kwargs = {
            'train_std': True,
        }, 
    )
)

importance_weighted = deep_dictionary_update(
    copy.deepcopy(log_policy),
    dict(
        description = {
            'name': 'importance-weighted',
        },
        inference_network_type = IWNeuralProcess, 
    )
)

scheduled = deep_dictionary_update(
    copy.deepcopy(log_policy),
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
    copy.deepcopy(log_policy),
    dict(
        description = {
            'name': 'scheduled-beta-reg',
        },
        trainer_kwargs = dict(
            beta_schedule = cyclic_schedule_regularization,
        )
    )
)



if __name__ == '__main__':
    # Set GPU device as indexed by nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # Choose configs to run
    configs = [
        log_policy,
        trainable_std,
        importance_weighted,
        scheduled,
        scheduled_regularized,
    ]
    
    for config in configs:
        experiment_name = config["environment_factory"].__name__ + "_" + config['description']['name']
        logger_kwargs = dict(
            log_dir=os.path.join("/data/bing/julius/encoder_training", experiment_name),
            snapshot_mode='last',
            snapshot_gap=25,
        )

    algorithm = init_algorithm(config, logger_kwargs=logger_kwargs)
    algorithm.train()