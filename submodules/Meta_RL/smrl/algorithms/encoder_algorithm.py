"""
This module contains the class ``MdpVaeAlgorithm`` which can train an encoder-
decoder architecture to predict MDP transitions and rewards. It serves as a 
stand-alone algorithm to (pre-)train the inference module. 

If not strictly required, you should use ``MetaRlAlgorithm`` instead. 
This algorithm (``MdpVaeAlgorithm``) can serve test purposes (e.g. find out which 
hyperparameters work well) and is more light-weight.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-03-06
"""

from tqdm import tqdm
import torch
from typing import Union, Tuple
from collections.abc import Iterable

from torch.optim.lr_scheduler import _LRScheduler

from rlkit.core.logging import Logger, logger as default_logger

from smrl.vae.mdpvae import MdpEncoder, MdpDecoder, MdpVAE
from smrl.data_management.vae_training_data import ContextTargetBuffer, ContextCollector
from smrl.trainers.vae import MdpVAETrainer
from smrl.utility import console_strings



class MdpVaeAlgorithm():
    """An algorithm for training an MdpEncoder based on batch data.

    Parameters
    ----------
    trainer : MdpVAETrainer
        Trainer for inference mechanisms
    data_collector : ContextCollector
        Data collector for contexts & prediction targets
    buffer : ContextTargetBuffer
        Buffer for contexts & prediction targets
    batch_size : int
        Batch size
    train_calls_per_epoch : int
        Number of training steps in each epoch. Each step corresponds to 
        one gradient step.
    samples_per_epoch : int
        Number of new context & target samples in each epoch.
    initial_samples : int
        Number of initial training samples collected before training starts.
        By default 0
    logger : rlkit.core.logging.Logger
        Logger which is used for logging. If no value is provided, the default
        logger from torch.rlkit.core.logging is used.
    lr_scheduler :_LRScheduler | Tuple[_LRScheduler, ...]
        Learning rate scheduler(s). The scheduler's function ``step()`` is
        called after each epoch. You can set up the scheduler with optimizers
        of the trainer before initializing the algorithm.
        By default ().
    """
    def __init__(
            self,
            trainer: MdpVAETrainer,
            data_collector: ContextCollector,
            buffer: ContextTargetBuffer,
            n_epochs: int,
            batch_size: int,
            train_calls_per_epoch: int,
            samples_per_epoch: int, 
            initial_samples: int = 0,
            logger: Logger = None,
            lr_scheduler: Union[_LRScheduler, Tuple[_LRScheduler, ...]] = None,
        ) -> None:
        self.trainer = trainer
        self.buffer = buffer
        self.collector = data_collector

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_calls_per_epoch = train_calls_per_epoch
        self.samples_per_epoch = samples_per_epoch
        self.initial_samples = initial_samples

        if lr_scheduler is None:
            lr_scheduler = ()
        if not isinstance(lr_scheduler, Iterable):
            lr_scheduler = (lr_scheduler, )
        self.lr_scheduler: Tuple[_LRScheduler, ...] = lr_scheduler

        self.logger = logger if logger is not None else default_logger

    def training_mode(self, mode: bool):
        for net in self.trainer.networks:
            net.train(mode)

    def train(self):
        console_strings.print_to_terminal("Collecting initial samples ...")
        self.training_mode(False)
        initial_samples = self.collector.collect_data(self.initial_samples)
        self.buffer.add_samples(*initial_samples)
        
        for epoch in range(self.n_epochs):
            self._start_epoch()
            self._train()
            self._end_epoch(epoch)

    def to(self, device: torch.device):
        for net in self.trainer.networks:
            net.to(device)

    def _train(self):

        console_strings.print_to_terminal("Collecting new samples ...")
        self.training_mode(False)
        new_samples = self.collector.collect_data(self.samples_per_epoch)
        self.buffer.add_samples(*new_samples)

        console_strings.print_to_terminal("Training ...")
        self.training_mode(True)
        for step in tqdm(range(self.train_calls_per_epoch), disable=(not console_strings.verbose)):
            context, target = self.buffer.random_batch(self.batch_size)
            self.trainer.train({
                'context': context,
                'target': target
            })
        self.training_mode(False)

    def _start_epoch(self):
        pass

    def get_diagnostics(self):
        stats = self.trainer.get_diagnostics()
        buffer_stats = self.buffer.get_diagnostics()
        for key, value in buffer_stats.items():
            stats['buffer/' + key] = value
        for i, scheduler in enumerate(self.lr_scheduler):
            stats[f'scheduler {i}/learning rate'],  = scheduler.get_lr()
        return stats

    def _end_epoch(self, epoch):
        # Scheduler update
        for scheduler in self.lr_scheduler:
            scheduler.step()

        # Save parameters
        snapshot = self.trainer.get_snapshot()
        self.logger.save_itr_params(epoch, snapshot)

        # Logging
        self.logger.record_dict(self.get_diagnostics())
        self.logger.record_tabular('Epoch', epoch)
        self.logger.dump_tabular(with_prefix=True, with_timestamp=True)
