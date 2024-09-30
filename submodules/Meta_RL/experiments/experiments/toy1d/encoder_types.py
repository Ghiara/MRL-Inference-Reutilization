from configs.environment_factory import toy1d_rand
from experiments.config_modules.inference import *
from experiments.config_modules.policy import policy_networks
from experiments.config_modules.algorithm import *
from experiments.config_modules.exploration import *

base_config = dict(

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 2,
    context_size = 5,

    **mlp_decoder(),
    **policy_networks(),
    **training_config(
        inference_lr=3e-3,
        num_epochs=5_000,
        prediction_target_size=2,
        num_eval_paths_per_epoch=5,
        num_expl_paths_per_epoch=5,
        num_inference_paths_per_epoch=5,
        num_inference_trains_per_train_loop=10,
        num_policy_trains_per_train_loop=2,
    ),
    **multi_random_memory_policy(
        action_update_interval = 5, 
        mean_std_range = (1e-5, 1e-1), 
        std_mean_range = (1e-5, 1e-1), 
        S = 1e-5, 
        inference_only=True
    ),
)


""" Pair encoder """
pair_encoder_np = dict(
    description = {
        "name": "pair-encoder-np"
    },

    **base_config,

    **pair_encoder(hidden_sizes=(32,32,32,32)),
    **neural_process(),
)

pair_encoder_vae = dict(
    description = {
        "name": "pair-encoder-vae"
    },

    **base_config,

    **pair_encoder(hidden_sizes=(32,32,32,32)),
    **mdp_vae(),
)


""" MLP encoder """
mlp_encoder_vae = dict(
    description = {
        "name": "mlp-encoder-vae"
    },

    **base_config,
    
    **mlp_encoder(),
    **mdp_vae(),
)


""" GRU encoder """
gru_encoder_np = dict(
    description = {
        "name": "gru-encoder-np"
    },

    **base_config,

    **gru_encoder(),
    **neural_process(),
)

gru_encoder_vae = dict(
    description = {
        "name": "gru-encoder-vae"
    },

    **base_config,

    **gru_encoder(),
    **mdp_vae(),
)


""" Attention encoder """
attention_encoder_np = dict(
    description = {
        "name": "attention-encoder-np"
    },

    **base_config,

    **attention_encoder(self_attention_layers=0),
    **neural_process(),
)
attention_encoder_np['inference_trainer_kwargs']['lr'] = 1e-4

attention_encoder_vae = dict(
    description = {
        "name": "attention-encoder-vae"
    },

    **base_config,

    **attention_encoder(self_attention_layers=0),
    **mdp_vae(),
)
attention_encoder_vae['inference_trainer_kwargs']['lr'] = 1e-4

if __name__ == "__main__":
    from experiments.experiments.runner import run_experiment

    configs = [
        pair_encoder_np,
        pair_encoder_vae,
        mlp_encoder_vae,
        gru_encoder_np,
        gru_encoder_vae,
        attention_encoder_np,
        attention_encoder_vae,
    ]

    for config in configs:
        config['environment_factory'] = toy1d_rand
        run_experiment(
            config,
            gpu="4",
            multithreading=True,
            log_dir='/data/bing/julius/experiments/toy1d_encoders'
        )