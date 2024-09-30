"""
This file serves as an example how to instantiate an algorithm instance
from a configuration file and run it afterwards.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-04-06
"""

import os
import torch
import ray
from datetime import datetime
import pytz

from smrl.experiment.experiment_setup import setup_experiment
from configs.base_configuration import config
# from configs.base_configuration_bigger_envs import config
from configs.environment_factory import toy1d_rand, toy1d


# Environment
config['environment_factory'] = toy1d


# GPU available?
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    # Makes sure the listing is in the same order as with nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"    # Set to "" if you do not want to use GPUs
print(f"GPU available: {'Yes' if torch.cuda.is_available() else 'No'}")

# Multithreading
os.environ["MULTITHREADING"] = "False"   # Set to "False" to not use multithreading
if os.environ["MULTITHREADING"] == "True":
    ray.init(num_cpus=12)


# Setup experiment, modify logging parameters
# experiment_name = f"{config['environment_factory'].__name__}_{config['description']['name']}"
experiment_name = "experiments_thesis/step1_biggerNN_-10_10/"
config['algorithm_kwargs']['num_epochs'] = 5_000
current_time = datetime.now().astimezone(pytz.timezone('Europe/Berlin'))
logger_kwargs = {
    'log_dir': f"./data/{experiment_name}_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}",
    'snapshot_mode': 'gap_and_last',
    'snapshot_gap': 10,
}
algorithm, description = setup_experiment(
    experiment_name, 
    config,
    # path_to_weights='/home/ubuntu/juan/Meta-RL/data/experiments_thesis/step1_biggerNN_-10_10/_2024-05-15_09-16-38',
    logger_kwargs=logger_kwargs
)


# RUN EXPERIMENT
algorithm.train()
