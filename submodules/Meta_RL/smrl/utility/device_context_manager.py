import torch
from abc import abstractmethod
from typing import Union, List, Any
import gtimer

import rlkit.torch.pytorch_util as ptu


class MovableObject():
    """An interface which implements the method ``to(device)``.
    """
    @abstractmethod
    def to(self, device: torch.device):
        raise NotImplementedError

class DeviceContext(object):
    """A context manager which changes ``ptu.device`` to a target device and
    which moves all passed modules to this device.

    When the context is left, all changes are reversed, i.e. ``ptu.device`` is
    set to the original device and all modules are moved to this device.

    Device movement times are logged by ``gtimer`` (key ``'moving between devices'``).

    Parameters
    ----------
    target_device : torch.device
        Device which should be used in this context
    modules : Union[MovableObject, List[MovableObject]]
        Modules which need to be moved to the new device while the context is 
        active.
    verbose : bool, optional
        If True, device movements are printed to the console, by default False
    """
    def __init__(
        self, 
        target_device: torch.device, 
        modules: Union[MovableObject, 
        List[MovableObject]], 
        verbose: bool = False
    ):
        self._original_device = ptu.device
        self._target_device = target_device
        if not isinstance(modules, list):
            modules = [modules]
        self._registered_modules = modules
        self.verbose = verbose
    
    def __enter__(self) -> torch.device:
        gtimer.blank_stamp()
        ptu.device = self._target_device
        for module in self._registered_modules:
            module.to(ptu.device)
        if self.verbose: print(f'Moved to device {ptu.device}')
        gtimer.stamp('moving between devices', unique=False)
        return ptu.device

    def __exit__(self, *args, **kwargs):
        gtimer.blank_stamp()
        ptu.device = self._original_device
        for module in self._registered_modules:
            module.to(ptu.device)
        if self.verbose: print(f'Moved to device {ptu.device}')
        gtimer.stamp('moving between devices', unique=False)
