from typing import Dict, Any

from smrl.data_management.replay_buffers import TrajectoryReplayBuffer, MultiTaskReplayBuffer


def replay_buffers() -> Dict[str, Any]:
    """
    Returns a config of default replay buffers for policy and inference training
    """
    return dict(
        path_collector_kwargs = {},
        replay_buffer_type = TrajectoryReplayBuffer,
        replay_buffer_kwargs = {
            'max_path_number': 1_000,
            'max_sub_size': 300,
        },
        inference_replay_buffer_type = MultiTaskReplayBuffer,
        inference_replay_buffer_kwargs = {
            'max_replay_buffer_size': 50_000,
            'max_sub_size': 2_500,
            'randomize_targets': True,
            'randomize_contexts': False,
        }
    )

def inference_trainer_config(inference_lr: float = 3e-4) -> Dict[str, Any]:
    """
    Returns a default configuration of inference trainer arguments
    """
    return dict(
        inference_trainer_kwargs = dict(
            lr=inference_lr,        # Learning rate for the decoder & encoder networks
            n_latent_samples=32,    # Number of latent samples for Monte-Carlo estimation
            clipping=None,
        )
    )

def policy_trainer_config(
        policy_lr: float = 1e-4, 
        qf_lr: float = 1e-4, 
        encoder_lr: float = None,
        use_automatic_entropy_tuning: bool = False,
    ) -> Dict[str, Any]:
    """
    Returns a default configuration of policy trainer arguments
    """
    return dict(
        policy_trainer_kwargs = dict(
            discount=0.99,          # Discount factor of the MDP
            policy_lr=policy_lr,    # Learning rate for the policy network
            qf_lr=qf_lr,            # Learning rate for the Q-function networks           
            encoder_lr=encoder_lr,  # Learning rate of the encoder in SAC
            use_automatic_entropy_tuning=use_automatic_entropy_tuning,
        )
    )


def algorithm_config(
    num_epochs: int = 5_000,
    max_path_length: int = 250,
    num_policy_trains_per_train_loop: int = 5,
    num_inference_trains_per_train_loop: int = 5,
    prediction_target_size: int = 5,
    num_eval_paths_per_epoch: int = 15,
    num_expl_paths_per_epoch: int = 10,
    num_inference_paths_per_epoch: int = 10,
    num_train_loops_per_epoch: int = 20,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Default configuration for MetaRLAlgorithm
    """
    return dict(
        algorithm_kwargs = dict(
            batch_size=batch_size,             # Batch size for batch-training (= number of samples which is passed to the trainers in each training step)
            num_epochs=num_epochs,             # Number of training epochs
            max_path_length=max_path_length,        # Maximum path length
            prediction_target_size=prediction_target_size,  # Size of the prediction targets for ELBO computation (number of transitions which the decoder needs to predict (reward & next observation))
            num_eval_paths_per_epoch=num_eval_paths_per_epoch,    # Number of trajectories used for evaluation
            num_expl_paths_per_epoch=num_expl_paths_per_epoch,    # Number of trajectories added to the policy replay buffer in each train loop
            num_inference_paths_per_epoch=num_inference_paths_per_epoch,   # Number of trajectories added to the inference replay buffer in each train loop
            num_inference_trains_per_train_loop=num_inference_trains_per_train_loop,  # Number of inference training steps in each train loop
            num_policy_trains_per_train_loop=num_policy_trains_per_train_loop, # Number of policy training steps in each train loop
            num_train_loops_per_epoch=num_train_loops_per_epoch,    # Number of train loops per epoch
        )
    )


def training_config(
    policy_lr: float = 1e-4,
    qf_lr: float = 1e-4,
    inference_lr: float = 3e-4,
    num_epochs: int = 5_000,
    max_path_length: int = 250,
    prediction_target_size: int = 5,
    use_automatic_entropy_tuning: bool = False,
    num_policy_trains_per_train_loop: int = 5,
    num_inference_trains_per_train_loop: int = 5,
    num_eval_paths_per_epoch: int = 15,
    num_expl_paths_per_epoch: int = 10,
    num_inference_paths_per_epoch: int = 10,
    num_train_loops_per_epoch: int = 20,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Standard parameters for training algorithm, includes configurations for
    - Replay buffers (``replay_buffers()``)
    - Inference training (``inference_trainer_config()``)
    - Policy training (``policy_trainer_config()``)
    - MetaRLAlgorithm (``algorithm_config()``)
    """
    return dict(
        **replay_buffers(),
        **inference_trainer_config(inference_lr=inference_lr),
        **policy_trainer_config(policy_lr=policy_lr, qf_lr=qf_lr, 
                                use_automatic_entropy_tuning=use_automatic_entropy_tuning),
        **algorithm_config(
            num_epochs=num_epochs, 
            max_path_length=max_path_length, 
            prediction_target_size=prediction_target_size,
            num_policy_trains_per_train_loop=num_policy_trains_per_train_loop,
            num_inference_trains_per_train_loop=num_inference_trains_per_train_loop,
            num_eval_paths_per_epoch=num_eval_paths_per_epoch,
            num_expl_paths_per_epoch=num_expl_paths_per_epoch,
            num_inference_paths_per_epoch=num_inference_paths_per_epoch,
            num_train_loops_per_epoch=num_train_loops_per_epoch,
            batch_size=batch_size,
        ),
    )


def only_policy_training(
    policy_lr: float = 1e-4,
    qf_lr: float = 1e-4,
    num_epochs: int = 5_000,
    max_path_length: int = 250,
    use_automatic_entropy_tuning: bool = False,
) -> Dict[str, Any]:
    config = training_config(policy_lr=policy_lr, qf_lr=qf_lr, 
                             num_epochs=num_epochs, 
                             max_path_length=max_path_length,
                             use_automatic_entropy_tuning=use_automatic_entropy_tuning)
    config['algorithm_kwargs']['num_inference_paths_per_epoch'] = 0
    config['algorithm_kwargs']['num_inference_trains_per_train_loop'] = 0
    return config


def only_inference_training(
    inference_lr: float = 3e-4,
    num_epochs: int = 5_000,
    max_path_length: int = 250,
    prediction_target_size: int = 5,
) -> Dict[str, Any]:
    config = training_config(
        inference_lr=inference_lr, 
        num_epochs=num_epochs, 
        max_path_length=max_path_length, 
        prediction_target_size=prediction_target_size,
    )
    config['algorithm_kwargs']['num_expl_paths_per_epoch'] = 0
    config['algorithm_kwargs']['num_eval_paths_per_epoch'] = 0
    config['algorithm_kwargs']['num_policy_trains_per_train_loop'] = 0
    return config
