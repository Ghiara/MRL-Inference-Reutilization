import torch
from smrl.utility.ops import deep_dictionary_update

from ..base_configuration import config as base_config


config = dict(
    description = {
        'name': 'SGD-optimizer',
        'file': __file__,
        'variant': 'SGD optimizer',
    },
    policy_trainer_kwargs = {
        'optimizer_class': torch.optim.SGD,
    }
)

config = deep_dictionary_update(base_config, config)
