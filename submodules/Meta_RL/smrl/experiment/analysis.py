"""
This module contains convenience functions for result analysis:
- ``load_results()``
- ``load_models()``
- ``sample_latent_encodings()``

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-02
"""

from pathlib import Path
import pandas as pd
import json

from typing import Dict, List, Tuple

from smrl.utility.ops import ensure_importable_entries
from .model_setup import init_networks, load_params
from configs.environment_factory import toy1d_domain_rand

from stable_baselines3 import SAC


def result_loader(paths: List[str], itr: int = None) -> Tuple[str, Dict, Dict, pd.DataFrame]:
    """Efficiently load multiple results from a list of dictionaries.
    For more information, see ``load_results()``.

    This function implements the generator framework such that only one
    result is loaded at a time (memory efficiency!).

    Generators: https://realpython.com/introduction-to-python-generators/

    Parameters
    ----------
    paths : List[str]
        List of paths to load
    itr : int, optional
        Iteration which should be loaded. If no value is provided,
        the latest iteration (with the highest iteration number) will
        be selected, by default None

    Yields
    ------
    Tuple[str, Dict, Dict, pd.DataFrame]
        Name of the experiment,
        Results (models, environment, config, progress data),
        Configuration dictionary,
        Progress data
    """
    for p in paths:
        results = load_results(p, itr)
        config = results['config']
        progress_data = results['progress']
        name = Path(p).name
        yield name, results, config, progress_data


def load_results(path_to_data: str = "./data", itr: int = None, load_environments: bool = True, max_multiplier: float = None, min_multiplier: float = None) \
    -> Dict:
    """Load trained models from a data directory.

    The data directory should contain the following files.
    - 'itr_<nr.>.pkl': Serialized networks (encoder, decoder, policy) and environment
    - 'progress.csv': Training progress log

    Parameters
    ----------
    path_to_data : str, optional
        Path where data is stored, by default "./data"
    itr : int, optional
        Iteration which should be loaded. If no value is provided,
        the latest iteration (with the highest iteration number) will
        be selected, by default None
    load_environments : bool, optional
        Set to ``False`` to use new environments. By default ``True``

    Returns
    -------
    Dict[Any]
        Dictionary with entries
        - 'epoch': Epoch number of trained models, if available
        - 'config': Configuration dictionary
        - 'encoder'
        - 'decoder'
        - 'qf1'
        - 'qf2'
        - 'target_qf1'
        - 'target_qf2'
        - 'policy'
        - 'expl_env'
        - 'eval_env'
        - 'inference_expl_policy'
        - 'progress': Progress data (``pandas.DataFrame``),
        - 'models': Models as obtained from ``load_models()``,
    """

    trained_model_path = Path(path_to_data)
    with open(trained_model_path.joinpath("variant.json")) as variant_file:
        config = json.load(variant_file)
    config = ensure_importable_entries(config)

    models = init_networks(config)
    models = load_params(models, path_to_data, itr, load_environments=load_environments)

    if not 'eval_env' in models.keys():
        print("Creating new environments...")
        if config['environment_factory'] == toy1d_domain_rand:
            models['expl_env'], models['eval_env'] = config['environment_factory'](config['multipliers'])
        else:
            models['expl_env'], models['eval_env'] = config['environment_factory']()
        models['expl_env'].set_meta_mode('train')
        models['eval_env'].set_meta_mode('test')

    progress = pd.read_csv(trained_model_path.joinpath("progress.csv"))

    # config['pretrained_policy'] = True
    # if 'pretrained_policy' in config:
    #     custom_objects = {}
    #     custom_objects = {
    #         "learning_rate": 0.0,
    #         "lr_schedule": lambda _: 0.0,
    #         "clip_range": lambda _: 0.0,
    #     }

    #     if config['pretrained_policy'] == True:
    #         models['policy'] = SAC.load("/home/ubuntu/juan/Meta-RL/rl-trained-agents/ppo/HalfCheetah-v3_1/HalfCheetah-v3.zip")

    return {
        'epoch': itr,
        'config': config,
        **models,
        'progress': progress,
        'models': models,
    }

