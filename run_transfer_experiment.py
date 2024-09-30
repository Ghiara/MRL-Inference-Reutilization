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
from configs.environment_factory import toy1d_rand
from experiments.experiments.encoder_transfer.toy1d_to_toy1d import huge_config


# Environment
config['environment_factory'] = toy1d_rand


# GPU available?
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    # Makes sure the listing is in the same order as with nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"    # Set to "" if you do not want to use GPUs
print(f"GPU available: {'Yes' if torch.cuda.is_available() else 'No'}")

# Multithreading
os.environ["MULTITHREADING"] = "True"   # Set to "False" to not use multithreading
if os.environ["MULTITHREADING"] == "True":
    ray.init(num_cpus=4)


# Setup experiment, modify logging parameters
experiment_name = "Multiplier with multi memory(exploration)_2023-11-19_11-47-48"
current_time = datetime.now().astimezone(pytz.timezone('Europe/Berlin'))
logger_kwargs = {
    'log_dir': f"./data/{experiment_name}_transfer",
    # 'log_dir': "./delete",
    'snapshot_mode': 'gap_and_last',
    'snapshot_gap': 10,
}
algorithm, description = setup_experiment(
    experiment_name, 
    huge_config,
    logger_kwargs=logger_kwargs
)


# RUN EXPERIMENT
algorithm.train()
