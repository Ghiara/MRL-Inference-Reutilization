"""
This module contains wrappers for multiple
- Trainers
- Data collectors
- Replay buffers
such that they can be passed as if they were a single instance.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-26
"""

import torch
from typing import List, Dict, Iterable
from collections import OrderedDict

from rlkit.core.trainer import Trainer
from rlkit.samplers.data_collector import DataCollector
from rlkit.data_management.replay_buffer import ReplayBuffer


class MultiTrainerWrapper(Trainer):
    """A Trainer class which wraps multiple Trainer instances.

    This class keeps track of multiple trainers and implements shared functions for
    - end_epoch
    - get_snapshot
    - get_diagnostics
    
    Instances of this class can be used to be passed on to TorchBatchRLAlgorithm.

    Parameters
    ----------
    trainers : Dict[str, Trainer]
        A dictionary of trainers with their assigned names as keys.
    """
    def __init__(self, trainers: Dict[str, Trainer]) -> None:
        super().__init__()
        self.trainers = trainers

    def end_epoch(self, epoch):
        for trainer in self.trainers.values():
            trainer.end_epoch(epoch)

    def get_snapshot(self):
        snapshot = {}
        for name, trainer in self.trainers.items():
            for key, value in trainer.get_snapshot().items():
                snapshot[name + "/" + key] = value
        return snapshot

    def get_diagnostics(self):
        stats = OrderedDict()
        for name, trainer in self.trainers.items():
            stats.update((name + "/" + key, value) for (key,value) in trainer.get_diagnostics().items())
        return stats

    def train(self, data):
        raise NotImplementedError("Please call train() on each trainer individually!")

    @property
    def networks(self) -> Iterable[torch.nn.Module]:
        """Networks of the trainers.

        Returns
        -------
        Iterable[torch.nn.Module]
            List of all networks
        """
        # This function is NOT derived from Trainer but from TorchTrainer.
        # It will break if self.trainers does not contain TorchTrainer objects.
        # However, it is expected that this function will only be called in a
        # TorchTrainer context (for which it is very useful).
        return [net for trainer in self.trainers.values() for net in trainer.networks]
    
class MultiCollectorWrapper(DataCollector):
    """A DataCollector class which wraps multiple DataCollector instances.

    This class keeps track of multiple data collectors and implements shared functions for
    - end_epoch
    - get_snapshot
    - get_diagnostics
    - get_epoch_paths
    
    Instances of this class can be used to be passed on to TorchBatchRLAlgorithm.

    Parameters
    ----------
    collectors : Dict[str, DataCollector]
        A dictionary of data collectors with their assigned names as keys.
    """
    def __init__(self, collectors: Dict[str, DataCollector]) -> None:
        super().__init__()
        self.collectors = collectors

    def end_epoch(self, epoch):
        for collector in self.collectors.values():
            collector.end_epoch(epoch)

    def get_snapshot(self):
        snapshot = {}
        for name, collector in self.collectors.items():
            for key, value in collector.get_snapshot().items():
                snapshot[name + "/" + key] = value
        return snapshot

    def get_diagnostics(self):
        stats = OrderedDict()
        for name, collector in self.collectors.items():
            stats.update((name + "/" + key, value) for (key,value) in collector.get_diagnostics().items())
        return stats

    def get_epoch_paths(self):
        epoch_paths = []
        for collector in self.collectors.values():
            epoch_paths.extend(collector.get_epoch_paths())
        return epoch_paths

    def collect_new_paths(self, *args, **kwargs):
        raise NotImplementedError("Please call collect_new_paths() on each collector individually!")

    def collect_new_steps(self, *args, **kwargs):
        raise NotImplementedError("Please call collect_new_steps() on each collector individually!")

    
class MultiBufferWrapper(ReplayBuffer):
    """A ReplayBuffer class which wraps multiple ReplayBuffer instances.

    This class keeps track of multiple buffers and implements shared functions for
    - end_epoch
    - get_snapshot
    - get_diagnostics
    
    Instances of this class can be used to be passed on to TorchBatchRLAlgorithm.

    Parameters
    ----------
    collectors : Dict[str, ReplayBuffer]
        A dictionary of replay buffers with their assigned names as keys.
    """
    def __init__(self, buffers: Dict[str, ReplayBuffer]) -> None:
        super().__init__()
        self.buffers = buffers

    def end_epoch(self, epoch):
        for buffer in self.buffers.values():
            buffer.end_epoch(epoch)

    def get_snapshot(self):
        snapshot = {}
        for name, buffer in self.buffers.items():
            for key, value in buffer.get_snapshot().items():
                snapshot[name + "/" + key] = value
        return snapshot

    def get_diagnostics(self):
        stats = OrderedDict()
        for name, buffer in self.buffers.items():
            stats.update((name + "/" + key, value) for (key,value) in buffer.get_diagnostics().items())
        return stats

    def random_batch(self, batch_size):
        raise NotImplementedError("Please call add_sample() on a specific buffer!")

    def add_sample(self, *args, **kwargs):
        raise NotImplementedError("Please call add_sample() on each buffer individually!")

    def add_path(self, *args, **kwargs):
        raise NotImplementedError("Please call add_path() on each buffer individually!")

    def add_paths(self, *args, **kwargs):
        raise NotImplementedError("Please call add_paths() on each buffer individually!")

    def terminate_episode(self):
        raise NotImplementedError("Please call terminate_episode() on each buffer individually!")

    def num_steps_can_sample(self, **kwargs):
        raise NotImplementedError("Please call num_steps_can_sample() on each buffer individually!")