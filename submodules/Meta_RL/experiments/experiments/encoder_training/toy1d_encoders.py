import os
import copy

from smrl.trainers.vae import MdpVAETrainer
from smrl.data_management.vae_training_data import ContextTargetTaskBuffer, ContextCollector
from smrl.vae.encoder_networks import GRUEncoder, AttentionEncoder, MlpEncoder
from smrl.vae.decoder_networks import SeparateMlpDecoder
from smrl.vae.mdpvae import MdpVAE, NeuralProcess
from smrl.policies.exploration import LogMultiRandomMemoryPolicy

from smrl.experiment.encoder_training import init_algorithm

from configs.environment_factory import toy1d_rand


mlp_encoder_vae = dict(
    description = {
        'name': 'mlp-encoder-vae',
    },

    environment_factory = toy1d_rand,

    inference_policy_type = LogMultiRandomMemoryPolicy,
    inference_policy_kwargs = dict(
        std_low=1e-4,
        std_high=1e-1,
        action_update_interval=10,
        sample_update_interval=False
    ),

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    # Inference mechanism
    encoder_type = MlpEncoder,
    encoder_kwargs = {
        'hidden_sizes': (32,32,32,32),
        'encoding_mode': 'mean_var'
    },
    decoder_type = SeparateMlpDecoder, 
    decoder_kwargs = {
        'hidden_sizes': [16, 16, ],
        'std_rew': 0.1,
        'std_obs': 0.1,
        'train_std': False,
    }, 
    inference_network_type = MdpVAE, 
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
        'n_epochs': 10_000,
        'batch_size': 128,
        'train_calls_per_epoch': 50,
        'samples_per_epoch': 20,
        'initial_samples': 5000,
    },
)

gru_encoder_vae = copy.deepcopy(mlp_encoder_vae)
gru_encoder_vae.update(
    dict(
        description = {
            'name': 'gru-encoder-vae',
        },
        encoder_type = GRUEncoder, 
        encoder_kwargs = {
            'hidden_size': 32,
            'num_layers': 4,
            'encoding_mode': 'mean_var',
        },
    )
)

gru_encoder_np = copy.deepcopy(gru_encoder_vae)
gru_encoder_np.update(
    dict(
        description = {
            'name': 'gru-encoder-np',
        },
        inference_network_type = NeuralProcess,
    )
)

attention_encoder_vae = copy.deepcopy(mlp_encoder_vae)
attention_encoder_vae.update(
    dict(
        description = {
            'name': 'attention-encoder-vae',
        },
        encoder_type = AttentionEncoder, 
        encoder_kwargs = {
            'self_attention_layers': 2,
            'query_layers': 2,
            'n_queries': 8,
            'num_heads': 2,
            'encoding_mode': 'mean_var',
        },
    )
)

attention_encoder_np = copy.deepcopy(attention_encoder_vae)
attention_encoder_np.update(
    dict(
        description = {
            'name': 'attention-encoder-np',
        },
        inference_network_type = NeuralProcess,
    )
)

attention_encoder_no_self_attention = copy.deepcopy(attention_encoder_vae)
attention_encoder_no_self_attention.update(
    dict(
        description = {
            'name': 'attention-encoder-vae-no-self-attention',
        },
        encoder_kwargs = {
            'self_attention_layers': 0,
            'query_layers': 2,
            'n_queries': 8,
            'num_heads': 2,
            'encoding_mode': 'mean_var',
        },
    )
)



if __name__ == '__main__':
    # Set GPU device as indexed by nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # Choose configs to run
    configs = [
        mlp_encoder_vae,
        gru_encoder_vae,
        gru_encoder_np,
        attention_encoder_vae,
        attention_encoder_np,
        attention_encoder_no_self_attention,
    ]

    for config in configs:
        experiment_name = config['description']['name']
        logger_kwargs = dict(
            log_dir=os.path.join("/data/bing/julius/encoder_training/toy1d", experiment_name),
            snapshot_mode='last',
            snapshot_gap=25,
        )

    algorithm = init_algorithm(config, logger_kwargs=logger_kwargs)
    algorithm.train()