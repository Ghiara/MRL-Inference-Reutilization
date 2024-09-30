""" 
This module contains implementations for MDP-Encoders:
- ``GRUEncoder``
- ``MlpEncoder``
- ``AttentionEncoder``
See also: mdpvae.py

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2022-12-31
"""


import torch
from typing import Union, Tuple, List, Type

from smrl.utility.distributions import DiagonalMultivariateNormal
from ..mdpvae import MdpEncoder
from .util import batched


class GRUEncoder(MdpEncoder):
    """An encoder which uses Gated Recurrent Units (GRU) to process MDP context data.
    The data is simply stacked on top of each other to form inputs to the network.

    Parameters
    ----------
    observation_dim : int
        Dimension of observations in the MDP.
    action_dim : int
        Dimension of actions.
    latent_dim : int
        Dimension of latent representations.
    encoding_dim : int
        Dimension of latent encodings (can be different from latent_dim, 
        e.g. if we use 'mean_var' as encoding_mode)
    hidden_size : int
        Hidden dimension of the GRU network.
    num_layers : int
        Number of hidden layers in the GRU network.
    batch_norm : bool, optional
        Adds a batch norm layer before the GRU network, by default False
    dropout : bool, optional
        Adds a dropout layer before the GRU network, by default False
    """
    supports_variable_sequence_length = True
    def __init__(
        self, 
        observation_dim: int,
        action_dim: int,
        latent_dim: int, 
        encoding_dim: int,
        hidden_size: int,
        num_layers: int,
        batch_norm: bool = False,
        dropout: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(observation_dim, action_dim, latent_dim, encoding_dim, *args, **kwargs)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        pre_gru_layers = []
        if batch_norm:
            pre_gru_layers.append(
                torch.nn.BatchNorm1d(2*observation_dim + action_dim + 2)
            )
        if dropout:
            pre_gru_layers.append(
                torch.nn.Dropout(p=0.5)
            )
        self._preprocess_network = torch.nn.Sequential(*pre_gru_layers)
        self._gru = torch.nn.GRU(
            input_size=2*observation_dim + action_dim + 2,  # sum of shapes: obs, next_obs, actions, reward, terminal
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self._mean_layer = torch.nn.Linear(hidden_size, latent_dim) # Outputs the mean vector of the distribution
        self._var_layer = torch.nn.Linear(hidden_size, latent_dim)  # Outputs the variance vector of the distribution
        # GRU network accepts inputs of shape (batch_size, sequence_length, input_dim)
        # OPTION: Consider using GRUCell instead to keep track of history without passing it every time.

    @batched
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        terminals: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:

        x = torch.cat((observations, actions, rewards, next_observations, terminals), dim=-1)
        batch_size = x.shape[0] if x.shape[0] != 0 else 1

        # If context is empty, return default distribution
        if x.nelement() == 0:
            mean = torch.zeros(torch.Size([batch_size, self.latent_dim])).to(self.device)
            var = torch.ones(torch.Size([batch_size, self.latent_dim])).to(self.device)
            return DiagonalMultivariateNormal(mean, var)

        batch_size, context_size, dims = x.shape[0], x.shape[1], x.shape[2:]
        # Preprocessing
        x = x.reshape(-1, *dims)
        x = self._preprocess_network(x)
        x = x.reshape(batch_size, context_size, *dims)

        # Pass input sequence through GRU
        h_0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device)  # Zeros are also default, so this is technically unnecessary
        h, _ = self._gru(x, h_0)
        h = h[:, -1, :] # We are only interested in the last time output

        # Apply output layers to obtain distribution parameters
        mean = self._mean_layer(h)
        var = self._var_layer(h).exp()
        return DiagonalMultivariateNormal(mean, var)