"""
This file provides encoders for specific use-cases, e.g. an oracle for Toy1D.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-19
"""

import torch

from smrl.vae.mdpvae import MdpEncoder
from smrl.utility.distributions import DiagonalMultivariateNormal
from smrl.vae.encoder_networks.util import batched
from rlkit.torch.distributions import Delta


class Toy1dOracle(MdpEncoder):
    """
    Oracle encoder for Toy1D which uses analytic computations for
    (almost) perfect predictions of the task.

    This encoder can be used as a reference for Toy1D encoders.
    """
    supports_variable_sequence_length = True
    
    def __init__(self, observation_dim: int, action_dim: int, latent_dim: int, encoding_dim: int, context_size: int = None, encoding_mode: str = 'sample', *args, **kwargs) -> None:
        super().__init__(observation_dim, action_dim, latent_dim, encoding_dim, context_size, encoding_mode, *args, **kwargs)
        assert latent_dim == 1, "Oracle for Toy1d returns 1d representations which are equal to the true task!"

    @torch.no_grad()
    @batched
    def forward(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, terminals: torch.Tensor) -> torch.distributions.Distribution:
        batch_size, sequence_len = observations.shape[0], observations.shape[1]
        if sequence_len <= 1:
            mean = torch.zeros(torch.Size([batch_size, self.latent_dim])).to(self.device)
            return DiagonalMultivariateNormal(mean, torch.ones_like(mean, device=self.device) * 1e-5)

        tasks = torch.zeros([batch_size, 1])
        
        for i, (obs, rew) in enumerate(zip(observations[..., 0], rewards)):
            min_idx = torch.argmin(obs)
            max_idx = torch.argmax(obs)
            min_obs, max_obs = obs[min_idx], obs[max_idx]
            dist1, dist2 = -rew[min_idx], -rew[max_idx]

            if min_obs == max_obs:
                tasks[i] = 0
                continue

            dist = max_obs - min_obs
            if dist2 > dist and dist2 > dist1:  # goal on the left
                tasks[i] = min_obs - dist1
            elif dist1 > dist and dist1 > dist2:    # goal on the right
                tasks[i] = max_obs + dist2
            else:   # goal in between observations
                tasks[i] = min_obs + dist1

        return DiagonalMultivariateNormal(tasks.to(self.device), torch.ones_like(tasks, device=self.device) * 1e-5)


