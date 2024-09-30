"""
This module contains implementations of the base classes ``ExplorationPolicy``
and ``ExplorationValueFunction``.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-01
"""

import torch
import numpy as np
from numpy import float32
from typing import List, Dict, Type, Union, Tuple

from rlkit.torch.distributions import Distribution, MultivariateDiagonalNormal, TanhNormal

from smrl.vae.encoder_networks.util import pad_sequence_length

from .base import StandardPolicy, ContextPolicy, ContextQFunction


class UniformPolicy(StandardPolicy):
    def __init__(self, action_dim: int, mean: torch.Tensor = torch.tensor(0.0), std: torch.Tensor = torch.tensor(1.0), *args, **kwargs) -> None:
        super().__init__(None, action_dim)
        self._mean = torch.ones(torch.Size([action_dim])) * mean
        self._std = torch.ones(torch.Size([action_dim])) * std
        self._action_dim = action_dim
        self.encoding_dim = 1
    def get_action(self, obs: np.ndarray, *inputs: np.ndarray, mode: str = 'sample', **kwargs):
        return np.array([np.random.uniform(-1,1)], dtype=float32), {}
    
    def forward(self, obs: torch.Tensor, *args, **kwarg) -> Distribution:
        dist = MultivariateDiagonalNormal(
            self._mean.expand([obs.shape[0], *self._mean.shape]),
            self._std.expand([obs.shape[0], *self._std.shape]),
        )
        return dist

class RandomPolicy(StandardPolicy):
    """A fully random policy.

    The actions are drawn from a multivariate Gaussian.

    Parameters
    ----------
    action_dim : int
        Dimension of the action space
    mean : torch.Tensor
        Mean
    std : torch.Tensor
        Standard deviation
    """
    def __init__(self, action_dim: int, mean: torch.Tensor = torch.tensor(0.0), std: torch.Tensor = torch.tensor(1.0), *args, **kwargs) -> None:
        super().__init__(None, action_dim)
        self._mean = torch.ones(torch.Size([action_dim])) * mean
        self._std = torch.ones(torch.Size([action_dim])) * std
        self._action_dim = action_dim
        self.encoding_dim = 1

    def forward(self, obs: torch.Tensor, *args, **kwarg) -> Distribution:
        dist = MultivariateDiagonalNormal(
            self._mean.expand([obs.shape[0], *self._mean.shape]),
            self._std.expand([obs.shape[0], *self._std.shape]),
        )
        return dist

    def to(self, device):
        super().to(device)
        self._mean.to(device)
        self._std.to(device)

class RandomMemoryPolicy(StandardPolicy):
    """A random policy which uses a memory to keep actions consistent over some
    number of steps.

    The actions are drawn from a multivariate Gaussian with 'current mean' and 
    'current_std'.
    These values are only updated after a specified number of calls and
    also drawn from a Gaussian/Exponential distribution with fixed parameters:

    ``action`` ~ N(``current_mean``, ``current_std``), (sample every step),\n
    ``current_mean`` ~ N(``mean``, ``mean_std``), (sample every ``update_interval`` steps)\n
    ``current_std`` ~ Exp(``std_mean``), (sample every ``update_interval`` steps)

    If ``sample_update_interval == True``, the action update interval is sampled as well:
    ``update_interval`` ~ Poisson(``action_update_interval``),
    (sample when steps since last update > ``update_interval``)

    Parameters
    ----------
    action_dim : int
        Dimension of the action space
    action_update_interval : int
        Number of steps after which the action mean will be udpated
    mean : Union[float, torch.Tensor], optional
        Fixed mean for drawing the current means.
        By default 0.0
    mean_std : Union[float, torch.Tensor], optional
        Fixed standard deviation for current means.
        By default 1.0
    std_mean : Union[float, torch.Tensor], optional
        Fixed mean for current standard deviations
    sample_update_interval : bool, optional
        If True, the update interval is sampled from a Poisson distribution.
        By default False
    """
    def __init__(
            self, 
            action_dim: int, 
            action_update_interval: int, 
            mean: Union[float, torch.Tensor] = 0.0, 
            mean_std: Union[float, torch.Tensor] = 1.0, 
            std_mean: Union[float, torch.Tensor] = 1.0, 
            sample_update_interval: bool = False,
            *args, **kwargs,
        ) -> None:
        super().__init__(None, action_dim)
        mean, mean_std, std_mean =  torch.ones([action_dim]) * mean, \
                                    torch.ones([action_dim]) * mean_std, \
                                    torch.ones([action_dim]) * std_mean
        mean, mean_std, std_mean =  mean.reshape(action_dim), \
                                    mean_std.reshape(action_dim), \
                                    std_mean.reshape(action_dim)
        
        self._steps_without_update = None
        self._update_interval = action_update_interval
        self._sample_update_interval = sample_update_interval

        self._mean_dist = torch.distributions.MultivariateNormal(mean, mean_std.diag_embed())
        self._std_dist = torch.distributions.Exponential(1 / std_mean)
        if sample_update_interval:
            self._interval_dist = torch.distributions.Poisson(action_update_interval)

        self._current_mean: torch.Tensor = None
        self._current_std: torch.Tensor = None

    @torch.no_grad()
    def forward(self, obs: torch.Tensor, *args, **kwarg) -> Distribution:
        if self._steps_without_update is None or self._steps_without_update >= self._update_interval:
            self._current_mean = self._mean_dist.sample()
            self._current_std = self._std_dist.sample()
            if self._sample_update_interval:
                self._update_interval = int(self._interval_dist.sample())
            self._current_mean.to(self.device)
            self._current_std.to(self.device)
            self._steps_without_update = 0
        dist = MultivariateDiagonalNormal(
            self._current_mean.expand(obs.shape[0], *self._current_mean.shape), # ensure that output is batched!
            self._current_std.expand(obs.shape[0], *self._current_std.shape),
        )
        self._steps_without_update += 1
        return dist

    def reset(self, *args, **kwargs):
        self._steps_without_update = None
        super().reset(*args, **kwargs)

    def __repr__(self):
        s = f"RandomMemoryPolicy with \n\t- distribution over means: {self._mean_dist.__repr__()} \n\t- distribution over std. dev.: {self._std_dist.__repr__()}"
        s += f"\n\t* current_mean = {self._current_mean}, \n\t* current_std = {self._current_std}, \n\t* update_interval = {self._update_interval}"
        return s


class MultiRandomMemoryPolicy(StandardPolicy):
    """A random policy which uses a memory to keep actions consistent over some
    number of steps. (c.f. ``RandomMemoryPolicy``)
    Every time that ``reset()`` is called, the mean and standard deviation of
    the policy change according to a distribution.

    NOTE: This policy can be understood as a collection of 
    multiple ``RandomMemoryPolicy`` policies.
    
    The action values are generated by the following process:

    ``action`` ~ N(``current_mean``, ``current_std``), (sample every step),\n
    ``current_mean`` ~ N(``mean``, ``mean_std``), (sample every ``update_interval`` steps)\n
    ``current_std`` ~ Exp(``std_mean``), (sample every ``update_interval`` steps),\n

    ``mean`` ~  N(``M``, ``S``), (sample every time that ``reset()`` is called),\n
    ``mean_std`` ~ Uniform(``mean_std_range``), (sample every time that ``reset()`` is called),\n
    ``std_mean`` ~ Uniform(``std_mean_range``), (sample every time that ``reset()`` is called)

    If ``sample_update_interval == True``, the action update interval is sampled as well:
    ``update_interval`` ~ Poisson(``action_update_interval``),
    (sample when steps since last update > ``update_interval``)

    Parameters
    ----------
    action_dim : int
        Dimension of the action space
    action_update_interval : int
        Number of steps after which the action mean will be udpated
    M : Union[float, torch.Tensor], optional
        Mean for drawing the means. By default 0
    S : Union[float, torch.Tensor], optional
        Standard deviation for drawing the means. By default 1.
    mean_std_range: Tuple[float, float], optional
        Range for drawing the variable ``mean_std``. By default (0.0, 1.0)
    std_mean_range: Tuple[float, float], optional
        Range for drawing the variables ``std_mean``. By default (0.0, 1.0)
    sample_update_interval : bool, optional
        If True, the update interval is sampled from a Poisson distribution.
        By default False
    """
    def __init__(
        self, 
        action_dim: int, 
        action_update_interval: int, 
        M: Union[float, torch.Tensor] = 0.0, 
        S: Union[float, torch.Tensor] = 1.0,
        mean_std_range: Tuple[float, float] = (0.0, 1.0),
        std_mean_range: Tuple[float, float] = (0.0, 1.0),
        sample_update_interval: bool = False,
        *args, **kwargs
    ) -> None:
        super().__init__(None, action_dim)
        self.action_update_interval = action_update_interval
        self.sample_update_interval = sample_update_interval

        self.M = torch.tensor(M) if M is not None else torch.zeros([action_dim])
        self.S = torch.ones([action_dim]) * torch.tensor(S) if S is not None else torch.ones([action_dim])
        self.mean_std_range = list(mean_std_range)
        self.std_mean_range = list(std_mean_range)
        if self.mean_std_range[0] == 0.0:
            self.mean_std_range[0] = (self.mean_std_range[1] - self.mean_std_range[0]) * 1e-5
        if self.std_mean_range[0] == 0.0:
            self.std_mean_range[0] = (self.std_mean_range[1] - self.std_mean_range[0]) * 1e-5

        self._mean_dist = MultivariateDiagonalNormal(self.M, self.S)
        self._mean_std_dist = torch.distributions.Uniform(*self.mean_std_range)
        self._std_mean_dist = torch.distributions.Uniform(*self.std_mean_range)

        self._random_policy: RandomMemoryPolicy = None
        self.reset()

    def reset(self, mean: torch.Tensor = None, mean_std: torch.Tensor = None, std_mean: torch.Tensor = None):
        mean = mean if mean is not None else self._mean_dist.sample()
        mean_std = mean_std if mean_std is not None else self._mean_std_dist.sample()
        std_mean = std_mean if std_mean is not None else self._std_mean_dist.sample()
        self._random_policy = RandomMemoryPolicy(
            action_dim=self.action_dim, 
            action_update_interval=self.action_update_interval, 
            mean=mean,
            mean_std=mean_std,
            std_mean=std_mean,
            sample_update_interval=self.sample_update_interval,
        )
        
    @torch.no_grad()
    def forward(self, obs: torch.Tensor, *args, **kwargs) -> Distribution:
        return self._random_policy.forward(obs, *args, **kwargs)

    def __repr__(self):
        s = "MultiRandomMemoryPolicy, currently active policy is: \n"
        s += self._random_policy.__repr__()
        return s


class LogMultiRandomMemoryPolicy(StandardPolicy):
    """A random policy which uses a memory to keep actions consistent over some
    number of steps. (c.f. ``RandomMemoryPolicy``)
    Every time that ``reset()`` is called, the mean and standard deviation of
    the policy change according to a distribution.

    NOTE: This policy can be understood as a collection of 
    multiple ``RandomMemoryPolicy`` policies.
    
    The action values are generated by the following process:

    ``action`` ~ N(``current_mean``, ``current_std``), (sample every step),\n
    ``current_mean`` ~ N(``mean``, ``std``), (sample every ``update_interval`` steps)\n
    ``current_std`` ~ Exp(``std``), (sample every ``update_interval`` steps),\n

    ``mean`` ~  N(``mean_mean``, ``std``), (sample every time that ``reset()`` is called),\n
    log(``std``) ~ Uniform(log(``std_low``), log(``std_high``)), (sample every time that ``reset()`` is called),\n

    If ``sample_update_interval == True``, the action update interval is sampled as well:
    ``update_interval`` ~ Poisson(``action_update_interval``),
    (sample when steps since last update > ``update_interval``)

    Parameters
    ----------
    action_dim : int
        Dimension of the action space
    action_update_interval : int
        Number of steps after which the action mean will be udpated
    std_low : float
        Lower value for standard deviations
    std_high : float
        Higher value for standard deviations
    mean_mean : Union[float, torch.Tensor], optional
        Mean of means (see ``RandomMemoryPolicy``). By default 0
    sample_update_interval : bool, optional
        If True, the update interval is sampled from a Poisson distribution.
        By default False
    """
    def __init__(
        self, 
        action_dim: int, 
        action_update_interval: int, 
        std_low: float,
        std_high: float,
        mean_mean: Union[float, torch.Tensor] = 0.0, 
        sample_update_interval: bool = False,
        *args, **kwargs
    ) -> None:
        super().__init__(None, action_dim)
        self.action_update_interval = action_update_interval
        self.log_std_low = torch.log(torch.tensor(std_low))
        self.log_std_high = torch.log(torch.tensor(std_high))
        self.log_std_dist = torch.distributions.Uniform(self.log_std_low, self.log_std_high)
        self.mean_mean = mean_mean
        self.sample_update_interval = sample_update_interval

        self._random_policy: RandomMemoryPolicy = None

    @torch.no_grad()
    def forward(self, obs: torch.Tensor, *inputs: torch.Tensor, **kwargs) -> Distribution:
        if self._random_policy is None:
            self.reset()
        return self._random_policy.forward(obs, *inputs, **kwargs)
    
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        log_std = self.log_std_dist.sample()
        self._random_policy = RandomMemoryPolicy(
            action_dim=self.action_dim,
            action_update_interval=self.action_update_interval,
            mean=self.mean_mean,
            mean_std=torch.exp(log_std),
            std_mean=torch.exp(log_std),
            sample_update_interval=self.sample_update_interval,
        )

    def __repr__(self):
        s = "LogMultiRandomMemoryPolicy, currently active policy is: \n"
        s += self._random_policy.__repr__()
        return s


class TanhExplorationPolicy(ContextPolicy):
    """This policy can be trained to explore the environment. It accepts contexts
    of recent transitions as a state-substitute and can base its transitions
    thereon.

    Parameters
    ----------
    context_size : int
        Number of transitions in the context.
    obs_dim : int
        Observation dimension
    act_dim : int
        Action dimension
    hidden_sizes : List[int]
        Sizes of the hidden layers
    activation_layer : Type[torch.nn.ReLU], optional
        Activation layer type, by default torch.nn.ReLU
    """
    def __init__(self, context_size: int, obs_dim: int, act_dim: int, hidden_sizes: List[int], activation_layer: Type[torch.nn.ReLU] = torch.nn.ReLU) -> None:
        super().__init__(context_size, obs_dim, act_dim)

        self.context_size = context_size
        self.concat_dims = 2*obs_dim + act_dim + 2
        
        sizes = [context_size * self.concat_dims] + hidden_sizes
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.BatchNorm1d(in_size))
            layers.append(torch.nn.Linear(in_size, out_size))
            layers.append(activation_layer())

        self.mlp = torch.nn.Sequential(*layers)

        self.mean_layer = torch.nn.Linear(hidden_sizes[-1], act_dim)
        self.std_layer = torch.nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, context: Dict[str, torch.Tensor], *args, **kwarg) -> Distribution:
        observations = context['observations']
        actions = context['actions']
        rewards = context['rewards']
        next_observations = context['next_observations']
        terminals = context['terminals']

        assert observations.ndim == 3, "Inputs must have three dimensions (batch_size, context_size, *)"
        batch_size, sequence_length = observations.shape[0], observations.shape[1]
        
        x = torch.concatenate([observations, actions, rewards, next_observations, terminals], dim=-1)
        if sequence_length < self.context_size:
            x = pad_sequence_length(self.context_size, x)

        x = self.mlp(x.reshape([batch_size, -1]))

        mean = self.mean_layer(x)
        std = self.std_layer(x).exp()

        return TanhNormal(mean, std)


class ConcatMlpValueFunction(ContextQFunction):
    """This Q-function can be trained to model the value of a ``TanhExplorationPolicy``.
    It accepts contexts of recent transitions as a state-substitute and can base 
    action-value thereon.

    Parameters
    ----------
    context_size : int
        Number of transitions in the context.
    obs_dim : int
        Observation dimension
    act_dim : int
        Action dimension
    hidden_sizes : List[int]
        Sizes of the hidden layers
    activation_layer : Type[torch.nn.ReLU], optional
        Activation layer type, by default torch.nn.ReLU
    """
    def __init__(self, context_size: int, obs_dim: int, act_dim: int, hidden_sizes: List[int], activation_layer: Type[torch.nn.ReLU] = torch.nn.ReLU) -> None:
        super().__init__(context_size, obs_dim, act_dim)

        sizes = [context_size * (2*obs_dim + act_dim + 2) + act_dim] + hidden_sizes

        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.Linear(in_size, out_size))
            layers.append(activation_layer())
        layers.append(torch.nn.Linear(out_size, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, context: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        x = torch.concatenate(
            [   # concatenate transition data: 
                # (batch_size, sequence_length, *) for observation, action, ...
                # -> (batch_size, sequence_length, obs_dim + act_dim + obs_dim + 2)
                context['observations'], 
                context['actions'], 
                context['rewards'], 
                context['next_observations'], 
                context['terminals'],
            ],
            dim=-1
        )
        batch_size = x.shape[0]
        x = pad_sequence_length(self.context_size, x)
        x = x.reshape(batch_size, -1)   # concatenate along time dimension
        x = torch.concatenate([x, action], dim=-1)  # add action
        value = self.network(x)
        return value
