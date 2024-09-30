"""
This file contains important variable definitions for a training run and a 
function for initializing them (``setup_experiment()``).

For your own experiment, create a config dictionary which implements all keys
from the base dictionary below which are ``None`` or ``{}``. The other keys can
be changed to have extended control over the algorithm.

Use the ``setup_experiment()`` function to generate all networks and set up the 
trainers, data management, and algorithm. The returned algorithm can be used for
training by calling `algorithm.train()`.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-26
"""

from typing import Dict, Any, Tuple
from collections import OrderedDict
import copy

from smrl.data_management.replay_buffers import ContextReplayBuffer
from smrl.algorithms.meta_rl_algorithm import MetaRlAlgorithm
from smrl.environments.meta_env import MetaEnv
from ._algorithm_setup import setup_algorithm
from .model_setup import init_networks, load_params

from smrl.utility.ops import ensure_importable_entries, deep_dictionary_update
from smrl.utility.console_strings import print_to_terminal, warning, ok

from configs.environment_factory import toy1d_domain_rand


"""
Base configuration dictionary. Overwrite *every* required key.
"""

base_config = OrderedDict(

# === REQUIRED keys === (need to be provided to prevent crashing)

    description = {
        'name': 'Base configuration',
        'file': __file__,
        'variant': '-',
        'inference': '-',
        'policy training': '-',
    },

    environment_factory = None, # Function which returns exploration and evaluation environment: () -> (MetaEnv, MetaEnv)

    observation_dim = None, # Dimension of the observaions
    action_dim = None,      # Dimension of the actions
    latent_dim = None,      # Latent dimension (which is used in the Inference mechanism)
    encoding_dim = None,    # Encoding dimension (dimension of the encodings given to the policy). Can be different to latent_dim, e.g. if encoder.encoding_mode is equal to 'mean_var'
    context_size = None,    # Length of the context sequences from which the encodings are derived

    # Inference mechanism
    encoder_type = None,    # Encoder class or any other encoder generator function
    encoder_kwargs = {},    # Arguments for encoder instantiation
    decoder_type = None,    # Decoder class
    decoder_kwargs = {},    # Arguments for decoder instantiation
    inference_network_type = None,  # Inference mechanism class, e.g. MdpVAE
    inference_network_kwargs = {},  # Arguments for the inference network instantiation

    # Policy training
    qf_network_type = None, # Q-function network class
    qf_network_kwargs = {}, # Arguments for q-function network instantiation
    policy_type = None,     # Policy network class
    policy_kwargs = {},     # Policy network instantiation arguments


# === OPTIONAL keys === (can be left as they are without crashing)

    # Encoder decoration
    encoder_decorator_type = None,  # Decorator class for the encoder, see smrl > vae > encoder_decorators.
    encoder_decorator_kwargs = {},  # Arguments for the encoder decorator instantiation.

    # Exploration (rollout policy) (for on-policy tranining, do not override)
    expl_policy_type = None,        # Exploration policy for off-policy training (None -> on-policy training)
    expl_policy_kwargs = {},        # Arguments for exploration policy (off-policy training)
    inference_policy_type = None,   # Exploration policy for off-policy inference training (None -> on-policy training)
    inference_policy_kwargs = {},   # Arguments for exploration policy (inference training) (off-policy training)

    # Algorithm
    path_collector_kwargs = {
        'save_env_in_snapshot': False,
    },
    replay_buffer_type = ContextReplayBuffer,
    replay_buffer_kwargs = {
        'max_replay_buffer_size': 1000,
        'randomize_contexts': False,
        'randomize_targets': False,
        # Depending on your choice or replay buffer, 
        # you might need to add additional items.
    },
    inference_replay_buffer_type = None,    # Specify this to have a different replay buffer type for inference exploration data
    inference_replay_buffer_kwargs = {},    # Arguments for the inference replay buffer if ``inference_replay_buffer_type`` is not None.
    inference_trainer_kwargs = dict(
        lr=1e-2,                    # Learning rate for the decoder & encoder networks
        n_latent_samples=10,        # Number of latent samples for Monte-Carlo estimation
    ),
    policy_trainer_kwargs = dict(
        discount=0.99,              # Discount factor of the MDP
        policy_lr=1e-3,             # Learning rate for the policy network
        qf_lr=1e-3,                 # Learning rate for the Q-function networks
        use_automatic_entropy_tuning=False,
    ),
    algorithm_kwargs = dict(
        batch_size=100,             # Batch size for batch-training (= number of samples which is passed to the trainers in each training step)
        num_epochs=5_000,             # Number of training epochs
        max_path_length=250,        # Maximum path length
        prediction_target_size=10,  # Size of the prediction targets for ELBO computation (number of transitions which the decoder needs to predict (reward & next observation))
        num_eval_paths_per_epoch=10,    # Number of paths used for evaluation in each epoch
        num_expl_paths_per_epoch=10,    # Number of paths added to the replay buffer in each epoch
        num_inference_paths_per_epoch=10,   # Number of paths added to the inference replay buffer in each epoch
        num_policy_trains_per_train_loop=10, # Number of policy training steps in each train loop
        num_inference_trains_per_train_loop=10,  # Number of inference training steps in each train loop
        num_train_loops_per_epoch=10,    # Number of train loops per epoch
    ),

)   # End of base configuration dictionary


def setup_experiment(
        experiment_name: str,
        config: Dict[str, Any],
        logger_kwargs: Dict[str, Any] = None,
        path_to_weights: str = None,
        itr: int = None,
    ) -> Tuple[MetaRlAlgorithm, Dict[str, Any]]:
    """Set up an experiment described in the `config` dictionary.

    This function ...
    1) ... instantiates new or loads existing networks / models
    2) ... sets up trainers and data management
    3) ... sets up the training algorithm and logger

    You can directly use the returned algorithm for training by calling its
    ``train()`` function.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    config : Dict[str, Any]
        Dictionary with parameters
    expl_env : MetaEnv
        Environment for exploration
    eval_env : MetaEnv
        Environment for evaluation
    logger_kwargs : Dict[str, Any], optional
        Parameters for the logger functionality, by default None
    path_to_weights : str, optional
        Path to pretrained models ('params.pkl' or 'itr_<nr.>.pkl') if
        training should be continued, by default None
    itr : int, optional
        Controls which version of pretrained models is used and determines the
        starting epoch. It is recommended to use this argument if ``path_to_weights``
        is provided to make sure that epoch counting is correct.

    Returns
    -------
    Tuple[MetaRLAlgorithm, Dist[str,Any]]
        Training algorithm
        Dictionary with configuration parameters (may be updated with default values)
    """

    # Fill configuration with default parameters (if not provided)
    config_ = deep_dictionary_update(copy.deepcopy(base_config), config)

    if path_to_weights is not None and itr is None:
        print_to_terminal(warning("``itr`` was not provided although ``path_to_weights`` was. Epoch counting may be incorrect!"))

    config_ = ensure_importable_entries(config_)

    # 0) Environment setup
    expl_env: MetaEnv
    eval_env: MetaEnv
    if config_['environment_factory'] == toy1d_domain_rand:
        expl_env, eval_env = config_['environment_factory'](config_['multipliers'])
    else:
        expl_env, eval_env = config_['environment_factory']()
    expl_env.set_meta_mode('train')
    eval_env.set_meta_mode('test')

    # 1) Instantiate or load models
    models = init_networks(config_)
    if path_to_weights is not None:
        models = load_params(models, path_to_weights, itr)
        try:
            expl_env = models['expl_env']
            eval_env = models['eval_env']
            expl_env.set_meta_mode('train')
            eval_env.set_meta_mode('test')
            print_to_terminal(ok("Loaded old environments. Will continue training with these."))
        except KeyError:
            print_to_terminal(warning("Could not find previous environments. Will continue training with new environments."))
        itr = models['epoch'] + 1
    
    if itr is None:
        itr = 0
    
    # 2,3) Setup training algorithm
    algorithm = setup_algorithm(
        experiment_name=experiment_name,
        config=config_,
        models=models,
        expl_env=expl_env,
        eval_env=eval_env,
        logger_kwargs=logger_kwargs,
        itr=itr
    )

    return algorithm, config_
    

