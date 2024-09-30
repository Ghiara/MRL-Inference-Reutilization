"""
This module contains utility functions to set up loggers.

NOTE: The functions in this file use RLKIT's loggers. (https://github.com/rail-berkeley/rlkit)

Author(s): 
    Julius Durmann, based on RLKIT!
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-13
"""

import os
import json
from typing import Dict, Any

from rlkit.core.logging import Logger
from rlkit.launchers.launcher_util import dict_to_safe_json


def setup_logger(
        variant: Dict[str, Any],
        log_dir: str,
        text_log_file: str = "debug.log",
        variant_log_file: str = "variant.json",
        tabular_log_file: str = "progress.csv",
        snapshot_mode: str = "last",
        snapshot_gap: int = 1,
        log_tabular_only: bool = False,
) -> Logger:
    """
    Sets up a logger.
    
    Parameters
    ----------
    variant : Dict[str, Any]
        Configuration dictionary
    text_log_file: str
        Name of the log file
    variant_log_file:
        Name of the configuration dictionary file
    tabular_log_file:
        Name of the progress data file
    snapshot_mode: str
        Determines how often snapshots are saved, e.g. 'last'.
    log_tabular_only: bool
        Set to True to not print output to console (?)

    Returns
    -------
    Logger
        A logger object
    """

    logger = Logger()

    if variant is not None:
        logger.log("Variant:")
        logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
        variant_log_path = os.path.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = os.path.join(log_dir, tabular_log_file)
    text_log_path = os.path.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)
    logger._add_output(tabular_log_path, logger._tabular_outputs,
                        logger._tabular_fds, mode='a')
    for tabular_fd in logger._tabular_fds:
        logger._tabular_header_written.add(tabular_fd)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    return logger