import os
import copy

from smrl.trainers.vae import MdpVAETrainer
from smrl.data_management.vae_training_data import ContextTargetTaskBuffer, ContextCollector
from smrl.vae.encoder_networks import GRUEncoder, PairAggregationEncoder
from smrl.vae.decoder_networks import SeparateMlpDecoder
from smrl.vae.mdpvae import NeuralProcess
from smrl.policies.exploration import LogMultiRandomMemoryPolicy, RandomMemoryPolicy, MultiRandomMemoryPolicy

from smrl.experiment.encoder_training import init_algorithm
from smrl.utility.ops import deep_dictionary_update

from configs.environment_factory import *


gru = dict(
    description = {
        'name': 'gru',
    },

    environment_factory = toy1d_rand,

    inference_policy_type = MultiRandomMemoryPolicy,
    inference_policy_kwargs = dict(
        action_update_interval=25,
        S=1e-5, 
        mean_std_range=(1e-5,1e-1),
        std_mean_range=(1e-5,1e-1), 
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


pair_aggregation = copy.deepcopy(gru)
pair_aggregation.update(
    dict(
        description = {
            'name': 'pair-aggregation',
        },
        encoder_type = PairAggregationEncoder, 
        encoder_kwargs = {
            'hidden_sizes': [32, 32, 32, 32],
            'encoding_mode': 'mean_var',
        },  
    )
)


if __name__ == "__main__":
    # Set GPU device as indexed by nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # Choose configs to run
    configs = [
        gru,
        pair_aggregation
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