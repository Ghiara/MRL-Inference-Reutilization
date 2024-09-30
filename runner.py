"""
A runner file which can be called from a console.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-13

Usage:
    Run
    ```
    python runner_julius.py --help
    ```
    for additonal information about the arguments.
Example:
    ```
    python runner_julius.py "configs.environment_factory.toy1d" "configs.base_configuration.config"
    ```
"""

import os
import torch
import ray
from datetime import datetime
import pytz
from typing import Dict, Callable, Tuple

from smrl.experiment.experiment_setup import setup_experiment
import smrl.utility.console_strings as console_strings

from smrl.environments.meta_env import MetaEnv


def main(
    environment_factory: Callable[[], Tuple[MetaEnv, MetaEnv]], 
    config: Dict, 
    log_dir: str = "./data", 
    gpu: str = "", 
    multithreading: bool = True, 
    save_env: bool = False,
    path_to_weights: str = None, 
    itr: str = None,
    verbose: bool = True,
    temp_dir: str = None,
    log_epoch_gap: int = 50,
):
    
    config['environment_factory'] = environment_factory
    if 'path_collector_kwargs' in config.keys():
        config['path_collector_kwargs']['save_env_in_snapshot'] = save_env
    else:
        config['path_collector_kwargs'] = {'save_env_in_snapshot': save_env}
    environment_name = environment_factory.__name__

    console_strings.verbose = verbose

    # GPU available?
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu

    # MULTITHREADING
    os.environ["MULTITHREADING"] = str(multithreading)
    if os.environ["MULTITHREADING"] == "True":
        if not ray.is_initialized():
            print("Initializing ray ...")
            ray.init(num_cpus=4, log_to_driver=console_strings.verbose, _temp_dir=temp_dir)

    experiment_name = f"{environment_name}_{config['description']['name']}"

    # Print summary before running experiment
    print("\n--------------------------------------------------")
    print(f"Experiment name:   {experiment_name}")
    print(f"Environment:       {environment_name}")
    print(f"Configuration:     {config['description']['name']}")
    print(f"Logging directory: {log_dir}")
    print(f"GPU:               {gpu if gpu != '' else '<None>'}")
    print(f"Multithreading:    {multithreading}")
    print(f"Save environment:  {save_env}")
    print(f"Verbose:           {verbose}")
    if path_to_weights is not None:
        print(f"Path to weights:   {path_to_weights}")
    if itr is not None:
        print(f"Iteration:         {itr}")
    if temp_dir is not None:
        print(f"Temp. directory:   {temp_dir}")
    print("--------------------------------------------------\n")

    print(f"GPU available: {'Yes' if torch.cuda.is_available() else 'No'}")
    # Run experiment
    current_time = datetime.now().astimezone(pytz.timezone('Europe/Berlin'))
    logger_kwargs = {
        'log_dir': os.path.join(log_dir, f"{experiment_name}_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}"),
        'snapshot_mode': 'gap_and_last',
        'snapshot_gap': log_epoch_gap,
    }
    if path_to_weights is not None:
        logger_kwargs['log_dir'] = path_to_weights
    algorithm, _ = setup_experiment(experiment_name, config, logger_kwargs, path_to_weights, itr=itr)
    
    # RUN EXPERIMENT
    algorithm.train()


if __name__ == "__main__":
    import argparse
    import importlib
    from distutils.util import strtobool

    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("environment", help="Environment factory function, see 'configs/environment_factory.py'. Use python notation, e.g. \"configs.environment_factory.toy1d\"")
    arg_parser.add_argument("config", help="The config dictionary, use python notation, e.g. \"configs.base_configuration.config\"", type=str)
    arg_parser.add_argument("--log_dir", help="Log directory", default="data/")
    arg_parser.add_argument("--gpu", help="List of available GPUs (see nvidia-smi)", type=str, default="")
    arg_parser.add_argument("--multithreading", help="Enable/disable multithreading", type=strtobool, default="True")
    arg_parser.add_argument("--save_env", help="Enable/disable environment saving in log parameters", type=strtobool, default="False")
    arg_parser.add_argument("--weights", help="Path to pretrained weights", type=str, default=None)
    arg_parser.add_argument("--itr", help="Iteration of pretrained weights", type=int, default=None)
    arg_parser.add_argument("--verbose", help="Enable/disable (most of) the terminal outputs", type=strtobool, default="True")
    arg_parser.add_argument("--temp_dir", help="Temporary directory for ray", type=str, default=None)
    args = arg_parser.parse_args()    

    # Load environment factory function (import module and function)
    factory_name = args.environment.split(".")[-1]
    module_name = args.environment.removesuffix("." + factory_name)
    module = importlib.import_module(module_name)
    environment_factory = getattr(module, factory_name)
    
    # Load config dictionary (import config module and import config name)
    config_name = args.config.split(".")[-1]
    module_name = args.config.removesuffix("." + config_name)
    module = importlib.import_module(module_name)
    config = getattr(module, config_name)

    main(
        environment_factory=environment_factory,
        config=config, 
        log_dir=args.log_dir,
        gpu=args.gpu, 
        multithreading=bool(args.multithreading),
        save_env=bool(args.save_env),
        path_to_weights=args.weights, 
        verbose=bool(args.verbose),
        temp_dir=args.temp_dir,
    )
