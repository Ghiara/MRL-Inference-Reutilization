"""
This module contains implementations of the base class ``MetaRLPolicy``

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-26
"""

import torch
from torch import nn
from typing import List, Type
import numpy as np

from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy

from rlkit.torch.distributions import TanhNormal
from stable_baselines3 import SAC

from .base import MetaRLPolicy

LOG_STD_MIN = -20
LOG_STD_MAX = 20
    
class MakeDeterministic(MetaRLPolicy):
    """Wrapper for stochastic policies which makes the action sampling
    deterministic. The sampled action is based on the 'maximum likelihood
    estimate' (usually the mean) of the action distribution.

    Parameters
    ----------
    action_distribution_generator : MetaRLPolicy
        Distribution which should be wrapped
    """
    def __init__(
            self,
            stochastic_policy: MetaRLPolicy,
    ):
        super().__init__(stochastic_policy.obs_dim, stochastic_policy.encoding_dim, stochastic_policy.action_dim)
        self.stochastic_policy = stochastic_policy

    def forward(self, *args, **kwargs):
        dist = self.stochastic_policy.forward(*args, **kwargs)
        return dist


class MetaRLTanhGaussianPolicy(MetaRLPolicy):
    """A simple Meta-RL policy which concatenates
    observations with latent contexts and passes them through
    a multi-layer perceptron (MLP).

    Parameters
    ----------
    obs_dim : int
        The observation dimension.
    encoding_dim : int
        The latent dimension of the context variables.
    action_dim : int
        The action dimension.
    hidden_sizes : List[int]
        The hidden layer sizes of the MLP.
    activation_layers : Type[torch.nn.ReLU], optional
        Activation layer class, by default torch.nn.ReLU
    """
    def __init__(
            self,
            obs_dim: int,
            encoding_dim: int,
            action_dim: int,
            hidden_sizes: List[int],
            activation_layer: Type[torch.nn.ReLU] = torch.nn.ReLU,
            **kwargs
    ):
        super().__init__(obs_dim, encoding_dim, action_dim, **kwargs)

        sizes = [obs_dim + encoding_dim] + list(hidden_sizes)
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.Linear(in_size, out_size))
            layers.append(activation_layer())
        self._mlp = torch.nn.Sequential(*layers)
        self._mean_layer = torch.nn.Linear(out_size, action_dim)
        self._std_layer = torch.nn.Linear(out_size, action_dim)

    def forward(self, obs: torch.Tensor, encoding: torch.Tensor):
        x = torch.cat((obs, encoding), -1)
        h = self._mlp(x)
        mean = self._mean_layer(h)
        log_std = self._std_layer(h)
        std = torch.exp(torch.clip(log_std, LOG_STD_MIN, LOG_STD_MAX))
        return TanhNormal(mean, std)
    

class MetaRLTanhGaussianPolicyWithMaxAction(MetaRLPolicy):
    """A simple Meta-RL policy which concatenates
    observations with latent contexts and passes them through
    a multi-layer perceptron (MLP).

    Parameters
    ----------
    obs_dim : int
        The observation dimension.
    encoding_dim : int
        The latent dimension of the context variables.
    action_dim : int
        The action dimension.
    hidden_sizes : List[int]
        The hidden layer sizes of the MLP.
    activation_layers : Type[torch.nn.ReLU], optional
        Activation layer class, by default torch.nn.ReLU
    """
    def __init__(
            self,
            obs_dim: int,
            encoding_dim: int,
            action_dim: int,
            hidden_sizes: List[int],
            activation_layer: Type[torch.nn.ReLU] = torch.nn.ReLU,
            max_action: int = 1.0,
            **kwargs
    ):
        super().__init__(obs_dim, encoding_dim, action_dim, **kwargs)

        sizes = [obs_dim + encoding_dim] + list(hidden_sizes)
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.Linear(in_size, out_size))
            layers.append(activation_layer())
        self._mlp = torch.nn.Sequential(*layers)
        self._mean_layer = torch.nn.Linear(out_size, action_dim)
        self._std_layer = torch.nn.Linear(out_size, action_dim)
        self.max_action = max_action

    def forward(self, obs: torch.Tensor, encoding: torch.Tensor):
        x = torch.cat((obs, encoding), -1)
        h = self._mlp(x)
        mean = self._mean_layer(h)
        log_std = self._std_layer(h)
        std = torch.exp(torch.clip(log_std, LOG_STD_MIN, LOG_STD_MAX))
        return TanhNormal(mean, std, self.max_action)
    

class PretrainedCheetah(MetaRLPolicy):
    """A simple Meta-RL policy which concatenates
    observations with latent contexts and passes them through
    a multi-layer perceptron (MLP).

    Parameters
    ----------
    obs_dim : int
        The observation dimension.
    encoding_dim : int
        The latent dimension of the context variables.
    action_dim : int
        The action dimension.
    hidden_sizes : List[int]
        The hidden layer sizes of the MLP.
    activation_layers : Type[torch.nn.ReLU], optional
        Activation layer class, by default torch.nn.ReLU
    """
    def __init__(
            self,
            obs_dim: int = 20,
            encoding_dim: int = 2,
            action_dim: int = 6,
            hidden_sizes: List[int] = [256,256,256],
            activation_layer: Type[torch.nn.ReLU] = torch.nn.ReLU,
            **kwargs
    ):
        super().__init__(obs_dim, encoding_dim, action_dim, **kwargs)

        self.pretrained_weights = torch.load("/home/ubuntu/juan/Meta-RL/submodules/ppo/HalfCheetah-v3_1/policy.pth")

        input_dim = 17  # Input dimension
        hidden_dim = 256  # Hidden dimension
        output_dim = 6  
        self.features_extractor = nn.Flatten()
        self.pi_features_extractor = nn.Flatten()
        self.vf_features_extractor = nn.Flatten()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer1.weight.data = self.pretrained_weights['mlp_extractor.policy_net.0.weight']
        self.layer2.weight.data = self.pretrained_weights['mlp_extractor.policy_net.2.weight']
        self.layer1.bias.data = self.pretrained_weights['mlp_extractor.policy_net.0.bias']
        self.layer2.bias.data = self.pretrained_weights['mlp_extractor.policy_net.2.bias']

        self.network = nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.layer2,
            nn.ReLU()
        ).to('cpu')
        self.action_net = nn.Linear(hidden_dim, output_dim).to('cpu')
        self.std_dev = nn.Linear(hidden_dim, output_dim).to('cpu')
        self.action_net.weight.data = self.pretrained_weights['action_net.weight'].to('cpu')
        self.action_net.bias.data = self.pretrained_weights['action_net.bias'].to('cpu')

    def forward(self, obs: torch.Tensor, encoding: torch.Tensor = None):
        if isinstance(obs, torch.Tensor):
            x = obs
        else:
            x = torch.from_numpy(obs.astype(np.float32))
        h = self.network(x[..., 3:])
        a = self.action_net(h)
        log_std = self.std_dev(h)
        std = torch.exp(torch.clip(log_std, LOG_STD_MIN, LOG_STD_MAX))
        return TanhNormal(a, std)
    def get_action(self, obs:np.ndarray, encoding:torch.Tensor = None):
        return elem_or_tuple_to_numpy(self._actions_from_distribution(self.forward(obs), mode='sample')), {}
        

        return 
    def get_actions(self, obs:np.ndarray):
        pass

