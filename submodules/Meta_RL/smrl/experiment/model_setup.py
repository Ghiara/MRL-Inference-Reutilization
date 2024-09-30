"""
This module contains the functions ``init_networks()`` and ``load_params()``.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-26
"""
 
import os
import re
import torch
from typing import Dict, Any
from pathlib import Path

from smrl.policies.meta_policy import MakeDeterministic


def init_networks(config: Dict[str, Any]) -> Dict[str, torch.nn.Module]:
    """Initialize untrained networks according to a configuration dictionary.
    The networks can be used for training with ``MdpVAETrainer`` and ``MetaSACTrainer``.

    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary with configuration details. See 'config/__init__.py' (top)
        for details on required keys and values.
    env : MetaEnv
        Environment (for dimensions)

    Returns
    -------
    Dict[str, torch.nn.Module]
        Dictionary with networks:
        - 'encoder'
        - 'decoder'
        - 'inference_network'
        - 'qf_networks'
        - 'policy':                 Policy which is trained
        - 'expl_policy':            Policy which is used for policy training rollouts (can differ from policy -> off-policy training)
        - 'inference_expl_policy':  Policy which is used for inference training rollouts
        - 'eval_policy':            Policy which is used for performance evaluation
    """

    obs_dim = config['observation_dim']
    action_dim = config['action_dim']
    latent_dim = config['latent_dim']
    encoding_dim = config['encoding_dim']
    context_size = config['context_size']

    # >>> 1) Networks <<<

    # VAE
    encoder_kwargs = dict(
        observation_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        encoding_dim=encoding_dim,
        context_size=context_size,
    )
    encoder_kwargs.update(config['encoder_kwargs'])   # This input preprocessing allows to override input by the config dictionary.
    encoder = config['encoder_type'](**encoder_kwargs)

    if 'encoder_decorator_type' in config.keys() and config['encoder_decorator_type'] is not None:
        encoder = config['encoder_decorator_type'](
            encoder,
            observation_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            encoding_dim=encoding_dim,
            context_size=context_size,
            **config['encoder_decorator_kwargs'],
        )

    decoder = config['decoder_type'](
        latent_dim=latent_dim,
        observation_dim=obs_dim,
        action_dim=action_dim,
        **config['decoder_kwargs']
    )

    inference_network = config['inference_network_type'](encoder, decoder, **config['inference_network_kwargs'])

    # SAC networks
    qf1 = config['qf_network_type'](
        obs_dim=obs_dim, 
        act_dim=action_dim, 
        encoding_dim=encoding_dim,
        **config['qf_network_kwargs'],
    )
    qf2 = config['qf_network_type'](
        obs_dim=obs_dim, 
        act_dim=action_dim, 
        encoding_dim=encoding_dim,
        **config['qf_network_kwargs'],
    )
    target_qf1 = config['qf_network_type'](
        obs_dim=obs_dim, 
        act_dim=action_dim, 
        encoding_dim=encoding_dim,
        **config['qf_network_kwargs'],
    )
    target_qf2 = config['qf_network_type'](
        obs_dim=obs_dim, 
        act_dim=action_dim, 
        encoding_dim=encoding_dim,
        **config['qf_network_kwargs'],
    )
    policy = config['policy_type'](
        obs_dim=obs_dim,
        encoding_dim=encoding_dim,
        action_dim=action_dim,
        **config['policy_kwargs'],
    )

    # Policies for exploration (can be on-policy or off-policy)
    eval_policy = MakeDeterministic(policy)
    if config['expl_policy_type'] is not None:  # off-policy
        expl_policy = config['expl_policy_type'](
            obs_dim=obs_dim,
            encoding_dim=encoding_dim,
            action_dim=action_dim,
            **config['expl_policy_kwargs'],
        )
    else:   # on-policy
        expl_policy = policy
    if config['inference_policy_type'] is not None: # special inference training policy
        inference_expl_policy = config['inference_policy_type'](
            obs_dim=obs_dim,
            encoding_dim=encoding_dim,
            action_dim=action_dim,
            **config['inference_policy_kwargs'],
        )
    else:   # same exploration policy as for policy training
        inference_expl_policy = expl_policy

    return dict(
        encoder = encoder,
        decoder = decoder,
        inference_network = inference_network,
        qf1 = qf1,
        qf2 = qf2,
        target_qf1 = target_qf1,
        target_qf2 = target_qf2,
        policy = policy,
        eval_policy = eval_policy,
        expl_policy = expl_policy,
        inference_expl_policy = inference_expl_policy,
    )


def load_params(models: Dict[str, torch.nn.Module], path_to_data: str, itr: int = None, load_environments: bool = True) -> Dict[str, torch.nn.Module]:
    """Load trained models from parameter files in ``path_to_data``.
    
    Parameter files can be named
    - 'itr_<nr.>.pkl'
    - 'params.pkl'

    If no iteration number (``itr``) is provided, the function search for the most recent parameter
    file ('params.pkl' or 'itr_<nr.>.pkl' with heighest <nr.>).

    NOTE: Models are set to test mode, i.e. ``model.train(False)`` is called
    for every model. Additionally, all models are loaded to the cpu.

    Parameters
    ----------
    models : Dict[str, torch.nn.Module]
        Dictionary of initialized models
    path_to_data : str
        Path to directory where parameters are located.
    itr : int, optional
        Iteration number. If not provided, most recent parameters will be selected,
        by default None
    load_environments : bool, optional
        Set to ``False`` to skip environment loading.
        By default ``True``

    Returns
    -------
    Dict[str, torch.nn.Module]
        Dictionary of trained models.
    """

    trained_model_path = Path(path_to_data)

    # Try to find parameter file in directory
    params = None
    if itr is not None:
        # Load specified iteration
        params = torch.load(trained_model_path.joinpath(f"itr_{itr}.pkl"), map_location=torch.device('cpu'))
        print(f"Using weights of iteration {itr}")
    
    if params is None:
        # Try to load latest parameters ('params.pkl')
        if trained_model_path.joinpath('params.pkl').is_file():
            params = torch.load(trained_model_path.joinpath("params.pkl"), map_location=torch.device('cpu'))
            print(f"Using weights of file 'params.pkl'")
    
    if params is None:
        # Try to load latest iteration
        for file in os.listdir(path_to_data):
            if file.endswith(".pkl"):
                itr_ = int(re.split('_|\.', file)[1])
                if itr is None or itr_ > itr:
                    itr = itr_
        if itr is None:
            raise RuntimeError("Could not find weights. The files should be named 'params.pkl' or 'itr_<nr.>.pkl'.")
        params = torch.load(trained_model_path.joinpath(f"itr_{itr}.pkl"), map_location=torch.device('cpu'))
        print(f"Using weights of iteration {itr}")

    # Load model parameters from state dictionaries
    models['encoder'].load_state_dict(params['trainer/Inference trainer/encoder'])
    models['decoder'].load_state_dict(params['trainer/Inference trainer/decoder'])
    models['qf1'].load_state_dict(params['trainer/Policy trainer/qf1'])
    models['qf2'].load_state_dict(params['trainer/Policy trainer/qf2'])
    models['target_qf1'].load_state_dict(params['trainer/Policy trainer/target_qf1'])
    models['target_qf2'].load_state_dict(params['trainer/Policy trainer/target_qf2'])
    models['policy'].load_state_dict(params['trainer/Policy trainer/policy'])

    # Environments
    if load_environments:
        try:
            models['expl_env'] = params['exploration/Exploration/env']
            models['eval_env'] = params['evaluation/env']
        except KeyError:
            pass

    models['epoch'] = params['epoch']

    return models