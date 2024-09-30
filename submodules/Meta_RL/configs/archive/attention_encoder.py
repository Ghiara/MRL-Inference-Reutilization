import copy

from smrl.utility.ops import deep_dictionary_update
from ..base_configuration import config as config_
from smrl.vae.encoder_networks import SelfAttentionEncoder, AttentionEncoder
from smrl.vae.encoder_decorators.striding import StridedEncoder
from smrl.data_management.replay_buffers import TrajectoryReplayBuffer


config = dict(

    description = {
        'name': 'Attention-encoder',
        'file': __file__,
        'variant': 'Attention-encoder',
        'inference': 'Attention-encoder',
    },

    encoder_type=AttentionEncoder,
    encoder_kwargs = dict(
        n_queries = 16,
        num_heads = 4,
        self_attention_layers = 2,
        query_layers = 2,
        encoding_mode='sample',
    ),
)

config = deep_dictionary_update(copy.deepcopy(config_), config)

strided_config = dict(
    description = {
        'name': 'Strided-attention-encoder',
        'file': __file__,
        'variant': 'Strided-attention-encoder',
        'inference': 'Strided attention encoder',
    },

    context_size = 25,

    encoder_decorator_type=StridedEncoder,
    encoder_decorator_kwargs={
        'stride': 5,
    },
)

strided_config = deep_dictionary_update(copy.deepcopy(config), strided_config)

