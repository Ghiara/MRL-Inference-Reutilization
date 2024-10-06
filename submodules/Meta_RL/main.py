"""
This script calls the main-function of the runner script.
It is useful for debugging.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-22
"""

from run_toy_training import main

from configs.environment_factory import *
from configs.base_configuration import config

main(
    toy1d,
    config,
    log_dir="./data/test",
    multithreading=True,
    gpu="0",
    log_epoch_gap=5,
    # temp_dir="/data/bing/julius/tmp/ray",
)