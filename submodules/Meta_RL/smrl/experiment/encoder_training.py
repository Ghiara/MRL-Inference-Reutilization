from typing import Dict, Any, Tuple
import torch

from rlkit.torch import pytorch_util as ptu

from smrl.algorithms.encoder_algorithm import MdpVaeAlgorithm
from smrl.vae.mdpvae import MdpEncoder, MdpDecoder, MdpVAE
from smrl.policies.base import Policy
from smrl.utility.logging import setup_logger


def init_models(config: Dict[str, Any]) -> Tuple[MdpEncoder, MdpDecoder, MdpVAE, Policy]:
    """
    Intialize encoder network, decoder network, and MDP-VAE
    
    Returns
    -------
    encoder : MdpEncoder
        Encoder
    decoder : MdpDecoder
        Decoder
    vae : MdpVAE
        Variational autoencoder network (or derived instances)
    policy : Policy
        Policy for rollouts
    """

    encoder = config['encoder_type'](
        config['observation_dim'],
        config['action_dim'],
        context_size=config['context_size'],
        latent_dim=config['latent_dim'],
        encoding_dim=config['encoding_dim'],
        **config['encoder_kwargs'],
    )

    decoder = config['decoder_type'](
        latent_dim=config['latent_dim'],
        observation_dim=config['observation_dim'],
        action_dim=config['action_dim'],
        **config['decoder_kwargs'],
    )

    vae = config['inference_network_type'](encoder, decoder, **config['inference_network_kwargs'])

    policy = config['inference_policy_type'](
        config['action_dim'],
        **config['inference_policy_kwargs']
    )

    return encoder, decoder, vae, policy


def init_algorithm(config: Dict[str, Any], logger_kwargs: Dict[str, Any]) -> MdpVaeAlgorithm:
    """
    Set up a training algorithm for MDP-encoders.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary. Please refer to the implementation of this
        function for required keys.
    logger_kwargs : Dict[str, Any]
        Arugments for the logger, see ``smrl.utility.logging.setup_logger()``
    """

    env, _ = config['environment_factory']()

    _, _, vae, policy = init_models(config)


    collector = config['collector_type'](
        env, 
        context_size=config['context_size'],  
        policy=policy,
        **config['collector_kwargs'],
    )
    buffer = config['buffer_type'](**config['buffer_kwargs'])
    trainer = config['trainer_type'](vae, **config['trainer_kwargs'])

    logger = setup_logger(config, **logger_kwargs)

    algorithm = MdpVaeAlgorithm(
        trainer=trainer,
        data_collector=collector,
        buffer=buffer,
        logger=logger,
        **config['algorithm_kwargs'],
    )
    
    print(f"GPU available: {torch.cuda.is_available()}")
    ptu.set_gpu_mode(torch.cuda.is_available())
    algorithm.to(ptu.device)

    return algorithm