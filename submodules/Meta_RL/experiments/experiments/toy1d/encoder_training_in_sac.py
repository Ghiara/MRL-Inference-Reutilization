from configs.environment_factory import toy1d_rand
from experiments.config_modules.inference import mlp_decoder, neural_process, gru_encoder
from experiments.config_modules.policy import policy_networks
from experiments.config_modules.algorithm import *
from experiments.config_modules.exploration import *


on_policy_only_sac = dict(
    description = {
        "name": "on-policy-only-sac"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 1,
    context_size = 5,

    **gru_encoder(),
    **mlp_decoder(),
    **neural_process(),
    **policy_networks(),
    **replay_buffers(),
    **inference_trainer_config(),
    **policy_trainer_config(encoder_lr=3e-4),
    algorithm_kwargs = dict(
        batch_size=128,             # Batch size for batch-training (= number of samples which is passed to the trainers in each training step)
        num_epochs=5_000,             # Number of training epochs
        max_path_length=250,        # Maximum path length
        prediction_target_size=5,  # Size of the prediction targets for ELBO computation (number of transitions which the decoder needs to predict (reward & next observation))
        num_eval_paths_per_epoch=15,    # Number of trajectories used for evaluation
        num_expl_paths_per_epoch=10,    # Number of trajectories added to the policy replay buffer in each train loop
        num_inference_paths_per_epoch=0,   # Number of trajectories added to the inference replay buffer in each train loop
        num_inference_trains_per_train_loop=0,  # Number of inference training steps in each train loop
        num_policy_trains_per_train_loop=1, # Number of policy training steps in each train loop
        num_train_loops_per_epoch=100,    # Number of train loops per epoch
    )
)

on_policy = dict(
    description = {
        "name": "on-policy"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 1,
    context_size = 5,

    **gru_encoder(),
    **mlp_decoder(),
    **neural_process(),
    **policy_networks(),
    **replay_buffers(),
    **inference_trainer_config(),
    **policy_trainer_config(encoder_lr=3e-4),
    algorithm_kwargs = dict(
        batch_size=128,             # Batch size for batch-training (= number of samples which is passed to the trainers in each training step)
        num_epochs=5_000,             # Number of training epochs
        max_path_length=250,        # Maximum path length
        prediction_target_size=5,  # Size of the prediction targets for ELBO computation (number of transitions which the decoder needs to predict (reward & next observation))
        num_eval_paths_per_epoch=15,    # Number of trajectories used for evaluation
        num_expl_paths_per_epoch=10,    # Number of trajectories added to the policy replay buffer in each train loop
        num_inference_paths_per_epoch=10,   # Number of trajectories added to the inference replay buffer in each train loop
        num_inference_trains_per_train_loop=1,  # Number of inference training steps in each train loop
        num_policy_trains_per_train_loop=1, # Number of policy training steps in each train loop
        num_train_loops_per_epoch=100,    # Number of train loops per epoch
    )
)

off_policy = dict(
    description = {
        "name": "off-policy"
    },

    observation_dim = 1,
    action_dim = 1,
    latent_dim = 1,
    encoding_dim = 1,
    context_size = 5,

    **gru_encoder(),
    **mlp_decoder(),
    **neural_process(),
    **policy_networks(),
    **replay_buffers(),
    **inference_trainer_config(),
    **policy_trainer_config(encoder_lr=3e-4),
    algorithm_kwargs = dict(
        batch_size=128,             # Batch size for batch-training (= number of samples which is passed to the trainers in each training step)
        num_epochs=5_000,             # Number of training epochs
        max_path_length=250,        # Maximum path length
        prediction_target_size=5,  # Size of the prediction targets for ELBO computation (number of transitions which the decoder needs to predict (reward & next observation))
        num_eval_paths_per_epoch=15,    # Number of trajectories used for evaluation
        num_expl_paths_per_epoch=10,    # Number of trajectories added to the policy replay buffer in each train loop
        num_inference_paths_per_epoch=10,   # Number of trajectories added to the inference replay buffer in each train loop
        num_inference_trains_per_train_loop=1,  # Number of inference training steps in each train loop
        num_policy_trains_per_train_loop=1, # Number of policy training steps in each train loop
        num_train_loops_per_epoch=100,    # Number of train loops per epoch
    ),
    **multi_random_memory_policy(action_update_interval = 5, mean_std_range = (1e-5, 1e-1), std_mean_range = (1e-5, 1e-1), S = 1e-5, inference_only=True),
)


if __name__ == "__main__":
    from experiments.experiments.runner import run_experiment

    configs = [
        on_policy,
    ]

    for config in configs:
        config['environment_factory'] = toy1d_rand
        run_experiment(
            config,
            gpu="4",
            multithreading=True,
            log_dir='/data/bing/julius/experiments/toy1d_sac-encoder-train'
        )