from typing import Dict, Any, List
from collections.abc import Iterable

from smrl.vae.encoder_networks import *
from smrl.vae.decoder_networks import *
from smrl.vae.mdpvae import *


def mlp_encoder(
    hidden_sizes: Iterable[int] = (32, 32, 32, 32), 
    encoding_mode: str = 'mean_var'
) -> Dict[str, Any]:
    """
    MLP encoder config
    """
    return dict(
        encoder_type = MlpEncoder,
        encoder_kwargs = dict(
            hidden_sizes=hidden_sizes,
            encoding_mode=encoding_mode,
        ),
    )

def mlp_decoder(
    hidden_sizes: Iterable[int] = (16, 16),
    std_rew: float = 0.1,
    std_obs: float = 0.1,
    train_std: bool = False,
) -> Dict[str, Any]:
    """
    Configuration for a ``SeparateMlpDecoder`` network
    """
    return dict(
        decoder_type = SeparateMlpDecoder,
        decoder_kwargs = {
            'hidden_sizes': hidden_sizes,
            'std_rew': std_rew,
            'std_obs': std_obs,
            'train_std': train_std,
        }
    )

def gru_encoder(
    hidden_size: int = 32, 
    num_layers: int = 4, 
    encoding_mode: str = 'mean_var',
    batch_norm: bool = False,
    dropout: bool = False,
) -> Dict[str, Any]:
    """
    GRU encoder config
    """
    return dict(
        encoder_type = GRUEncoder,
        encoder_kwargs = dict(
            hidden_size=hidden_size,
            num_layers=num_layers,
            encoding_mode=encoding_mode,
            batch_norm=batch_norm,
            dropout=dropout,
        ),
    )

def pair_encoder(
    hidden_sizes: Iterable[int] = (16, 16),
    encoding_mode: str = 'mean_var',
) -> Dict[str, Any]:
    return dict(
        encoder_type = PairAggregationEncoder,
        encoder_kwargs = dict(
            hidden_sizes=hidden_sizes,
            encoding_mode=encoding_mode,
        ),
    )

def attention_encoder(
    n_queries: int = 16,
    num_heads: int = 4,
    self_attention_layers: int = 2,
    query_layers: int = 2, 
    encoding_mode: str = 'mean_var',
) -> Dict[str, Any]:
    """
    Attention encoder config
    """
    return dict(
        encoder_type = AttentionEncoder,
        encoder_kwargs = dict(
            n_queries=n_queries,
            num_heads=num_heads,
            self_attention_layers=self_attention_layers,
            query_layers=query_layers,
            encoding_mode=encoding_mode
        )
    )


def mdp_vae(
    beta: float = 1.0
) -> Dict[str, Any]:
    """
    MDP-VAE config
    """
    return dict(
        inference_network_type=MdpVAE,
        inference_network_kwargs = {
            'beta': beta,
        }
    )

def mdp_iwae(
    beta: float = 1.0
) -> Dict[str, Any]:
    """
    MDP-IWAE config
    """
    return dict(
        inference_network_type=MdpIWAE,
        inference_network_kwargs = {
            'beta': beta,
        }
    )

def neural_process(
    beta: float = 1.0
) -> Dict[str, Any]:
    """
    Neural Process config
    """
    return dict(
        inference_network_type=NeuralProcess,
        inference_network_kwargs = {
            'beta': beta,
        }
    )

def iw_neural_process(
    beta: float = 1.0
) -> Dict[str, Any]:
    """
    Imporatance-weighted Neural Process config
    """
    return dict(
        inference_network_type=IWNeuralProcess,
        inference_network_kwargs = {
            'beta': beta,
        }
    )