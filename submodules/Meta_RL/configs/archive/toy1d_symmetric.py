import copy
from smrl.vae.mdpvae import NeuralProcess, MdpVAE

from symmetric_networks.toy1d_encoder import Toy1dEncoder, Toy1dGRUEncoder
from symmetric_networks.toy1d_policy import Toy1dPolicy, Toy1dQFunction

from ..base_configuration import config as base_config


symmetric_mlp_config = copy.deepcopy(base_config)
symmetric_mlp_config.update(dict(
    description = {
        'name': 'Symmetric-Toy1d-EquivariantMlp',
        'file': __file__,
        'variant': 'Equivariant Mlp encoder',
    },

    # Encoder
    encoder_type = Toy1dEncoder,
    encoder_kwargs = {
        'hidden_sizes': [32,32,32,32],
        'encoding_mode': 'sample',
    },
    inference_network_type = MdpVAE,

    # Policy
    qf_network_type = Toy1dQFunction,
    qf_network_kwargs = {
        'hidden_sizes': [16, 16, 16,],
    },
    policy_type = Toy1dPolicy,
    policy_kwargs = {
        'hidden_sizes': [16, 16, 16,],
    },
))

symmetric_gru_config = copy.deepcopy(symmetric_mlp_config)
symmetric_gru_config.update(dict(
    description = {
        'name': 'Symmetric-Toy1d-EquivariantGRU',
        'file': __file__,
        'variant': 'Equivariant GRU encoder',
    },
    encoder_type = Toy1dGRUEncoder,
    encoder_kwargs = {
        'hidden_size': 32,
        'encoding_mode': 'sample',
    },
    inference_network_type = NeuralProcess,
))
