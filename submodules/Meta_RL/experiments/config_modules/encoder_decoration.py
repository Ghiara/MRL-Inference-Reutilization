import os
import json
from typing import Dict, Any, List, Type, Union, Callable

from smrl.vae.encoder_decorators import (
    EncoderDecorator, 
    PretrainedEncoder, 
    InputOutputDecorator, 
    InputMaskedEncoder,
    StridedEncoder,
)
from smrl.vae.encoder_decorators.multi_decorator import MultiDecorator
from smrl.utility.ops import ensure_importable_entries
from smrl.utility.console_strings import warning


def load_config_from_file(
    path_to_data: str,
    config_file: str = 'variant.json',
):
    """
    Load an encoder configuration from a json file.
    """
    config = {}
    with open(os.path.join(path_to_data, config_file), mode='r') as file:
        encoder_config = json.load(file)
        encoder_config = ensure_importable_entries(encoder_config)
        config['encoder_type'] = encoder_config['encoder_type']
        config['encoder_kwargs'] = encoder_config['encoder_kwargs']
    return config


def pretrained_encoder(
    path_to_data: str,
    state_dict_keyword: str,
    weight_file: str = 'params.pkl',
    config_file: str = 'variant.json',
    trainable: bool = False,
    load_config: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Configuration for a pretrained encoder.

    Please provide the path to data (not weights!), i.e. the directory where
    the weights ('params.pkl') and config file ('variant.json') of the encoder
    training is located.

    NOTE: This function automatically loads the configuration from the encoder
    weight path (file 'variant.json') and adds it to the configuration dictionary.
    Please do not set the encoder configuration manually in this case.
    To stop this, set ``load_config`` to ``False`` (may be required for multi_decorator).
    """

    config = dict(
        encoder_decorator_type = PretrainedEncoder,
        encoder_decorator_kwargs = {
            'path_to_weights': os.path.join(path_to_data, weight_file),
            'trainable': trainable,
            'state_dict_keyword': state_dict_keyword,
            **kwargs,
        },
    )

    if load_config:
        try:
            config.update(load_config_from_file(path_to_data, config_file))
        except OSError:
            print(warning("Could not find encoder configuration file. "
                        + "Please make sure that you configured the encoder "
                        + "manually or that the encoder configuration file is in "
                        + f"'{path_to_data}'"))

    return config

def io_modified_encoder(
    input_map: Callable = None,
    output_map: Callable = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    An encoder config for input / output modification (e.g. mapping from 
    ``Toy1dContinuous`` to ``Toy1D``).
    """
    return dict(
        encoder_decorator_type = InputOutputDecorator,
        encoder_decorator_kwargs = {
            'input_map': input_map,
            'output_map': output_map,
            **kwargs,
        }
    )

def input_masked_encoder(
    mask: List[str],
    **kwargs,
) -> Dict[str, Any]:
    """
    Configuration for an input-masked encoder
    """
    return dict(
        encoder_decorator_type = InputMaskedEncoder,
        encoder_decorator_kwargs = {
            'mask': mask,
            **kwargs,
        },
    )

def strided_encoder(
    stride: int,
    **kwargs,
) -> Dict[str, Any]:
    """
    Configuration for a strided encoder
    """
    return dict(
        encoder_decorator_type = StridedEncoder,
        encoder_decorator_kwargs = {
            'stride': stride,
            **kwargs,
        }
    )

def multi_decorator(
    *encoder_types_and_kwargs: Dict[str, Union[Type[EncoderDecorator], Dict[str, Any]]],
    **kwargs,
) -> Dict[str, Any]:
    """
    Configuration for a multi-encoder-decorator.

    Parameters
    ----------
    *encoder_types_and_kwargs : 
        Dictionaries with entries:
            encoder_decorator_type : Type[EncoderDecorator]
                Type of an encoder decorator
            encoder_decorator_kwargs : Dict[str, Any]
                Arguments for the encoder decorator
        The first item is the inner encoder decorator.
    **kwargs
        Additional arguments for the MultiDecorator, e.g. encoding mode
    """
    return dict(
        encoder_decorator_type = MultiDecorator,
        encoder_decorator_kwargs = dict(
            **kwargs,
            decorators_and_kwargs = [
                (
                    encoder['encoder_decorator_type'],
                    encoder['encoder_decorator_kwargs']
                )
                for encoder in encoder_types_and_kwargs
            ]
        )
    )