"""
This module contains the function ``setup_algorithm()``.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-26
"""

import os
import torch
from typing import Dict, Any

from rlkit.core.logging import logger
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

from smrl.environments.meta_env import MetaEnv
from smrl.algorithms.meta_rl_algorithm import MetaRlAlgorithm
from smrl.data_management.path_collector import MdpPathCollector, MultithreadedPathCollector
from smrl.data_management.rollout_functions import rollout_with_encoder
from smrl.trainers.vae import MdpVAETrainer
from smrl.trainers.meta_sac import MetaSACTrainer


def setup_algorithm(
        experiment_name: str,
        config: Dict[str, Any],
        models: Dict[str, torch.nn.Module],
        expl_env: MetaEnv,
        eval_env: MetaEnv,
        logger_kwargs: Dict[str, Any] = None,
        itr: int = 0,
    ) -> MetaRlAlgorithm:
    """Setup training algorithm, including path collectors, buffer, and trainers.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    config : Dict[str, Any]
        Configuration dictionary.
    models : Dict[str, torch.nn.Module]
        Model (network) dictionary
    expl_env : MetaEnv
        Exploration environment
    eval_env : MetaEnv
        Evaluation environment
    logger_kwargs : Dict[str, Any], optional
        Arguments for the logger, by default None
    itr : int, optional
        Start epoch / iteration, by default 0

    Returns
    -------
    MetaRlAlgorithm
        Training algorithm
    """

    # >>> 2) Trainers and data management <<<

    # Rollout utility
    path_collector_class = MultithreadedPathCollector if os.environ["MULTITHREADING"] == "True" else MdpPathCollector
    expl_path_collector = path_collector_class(
        expl_env,
        models['expl_policy'],
        rollout_fn=rollout_with_encoder(models['encoder'], config['context_size']),
        **config['path_collector_kwargs'],
    )
    eval_path_collector = path_collector_class(
        eval_env,
        models['eval_policy'],
        rollout_fn=rollout_with_encoder(models['encoder'], config['context_size']),
        **config['path_collector_kwargs'],
    )
    inference_path_collector = path_collector_class(
        expl_env,
        models['inference_expl_policy'],
        rollout_fn=rollout_with_encoder(models['encoder'], config['context_size']),
        **config['path_collector_kwargs'],
    )
    replay_buffer = config['replay_buffer_type'](
        env=expl_env,
        encoding_dim=config['encoding_dim'],
        **config['replay_buffer_kwargs'],
    )
    if config['inference_replay_buffer_type'] is None:
        inference_replay_buffer = config['replay_buffer_type'](
            env=expl_env,
            encoding_dim=config['encoding_dim'],
            **config['replay_buffer_kwargs'],
        )
    else:
        inference_replay_buffer = config['inference_replay_buffer_type'](
            env=expl_env,
            encoding_dim=config['encoding_dim'],
            **config['inference_replay_buffer_kwargs'],
        )

    # Trainers
    inference_trainer = MdpVAETrainer(
        vae=models['inference_network'],
        **config['inference_trainer_kwargs']
    )
    policy_trainer = MetaSACTrainer(
        env=expl_env,
        policy=models['policy'],
        encoder=models['encoder'],
        qf1=models['qf1'],
        qf2=models['qf2'],
        target_qf1=models['target_qf1'],
        target_qf2=models['target_qf2'],
        **config['policy_trainer_kwargs'],
    )

    # >>> 3) Algorithm and Logging <<<

    algorithm = MetaRlAlgorithm(
        policy_trainer=policy_trainer,
        inference_trainer=inference_trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        inference_data_collector=inference_path_collector,
        expl_replay_buffer=replay_buffer,
        inference_replay_buffer=inference_replay_buffer,
        context_size=config['context_size'],
        start_epoch=itr,
        **config['algorithm_kwargs'],
    )

    ptu.set_gpu_mode(torch.cuda.is_available())  # Train on GPU, if available


    # Logger setup (use default parameters for values which were not provided)
    if logger_kwargs is None:
        logger_kwargs = {}
    logger_kwargs_ = dict(
        log_dir = "./data",
        snapshot_mode='gap_and_last',
        snapshot_gap=10,
    )
    logger_kwargs_.update(logger_kwargs)
    logger.reset()
    setup_logger(experiment_name, variant=config, **logger_kwargs_)

    algorithm.to(ptu.device)
    print(
        f"Devices:\n"  \
        + f"\tptu:        {ptu.device}\n"\
        + f"\tEncoder:    {models['encoder'].device}\n" \
        + f"\tDecoder:    {models['decoder'].device}\n" \
        + f"\tVAE:        {models['inference_network'].device}\n" \
        + f"\tPolicy:     {next(models['policy'].parameters()).device}\n"
        + f"\tQ-function: {next(models['qf1'].parameters()).device}\n"
    )

    return algorithm
