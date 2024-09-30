from smrl.utility.ops import deep_dictionary_update
from smrl.vae.encoder_networks import GRUEncoder
from smrl.vae.encoder_decorators.io_modification import InputMaskedEncoder

from ..base_configuration import config as base_config

config = dict(
    description = {
        'name': 'Encoder-Input-Masking',
        'file': __file__,
        'variant': 'Mask inputs actions ad rewards to encoder',
    },
    encoder_type = InputMaskedEncoder,
    encoder_kwargs = dict(
        encoder_class = GRUEncoder,
        mask = ['actions', 'next_observations'],
        hidden_size=32,
        num_layers=4,
        encoding_mode='sample',
    ),
)

config = deep_dictionary_update(base_config, config)
