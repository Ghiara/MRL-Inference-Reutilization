"""
This module contains custom distributions (derived from torch.distributions.Distribution).

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2022-11-22
"""

import torch
# from torch.distributions import Distribution
from collections import OrderedDict

from rlkit.torch.distributions import Distribution
from torch.distributions import Normal
from torch.distributions import constraints

class DiagonalMultivariateNormal(Distribution):
    """A multivariate normal distribution with diagonal covariance matrix.

    This class may help to circumvent memory issues with MultivariateNormal (Covariance matrix can be very large!).
    The covariance vector's memory requirements scale linear in dimension, instead of quadratic.

    Parameters
    ----------
    loc : torch.Tensor
        Mean vector
    covariance_vector : torch.Tensor
        Vector of diagonal entries of the covariance
    """
    
    arg_constraints = {
        'loc': constraints.real_vector,
        'covariance_vector': constraints.real_vector,
        'covariance_vector': constraints.greater_than_eq(0),
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc: torch.Tensor, covariance_vector: torch.Tensor, batch_shape=..., event_shape=..., validate_args=None):
        assert loc.shape[-1] == covariance_vector.shape[-1], "loc and covariance_vector must have same shape[-1]!"
        self.loc = loc
        self.covariance_vector = covariance_vector
        self.dim = self.loc.shape[-1]
        mu, sigma = torch.zeros(1).to(loc.device), torch.ones(1).to(loc.device)
        self._normal = Normal(loc=mu.squeeze(), scale=sigma.squeeze())   # Normal distribution for sampling
        self._shape = loc.shape
        super().__init__(batch_shape, event_shape, validate_args)

    def __repr__(self):
        return f"DiagonalMultivariateNormal with mean:\n{self.mean}\nand variance:\n{self.variance}"

    @property
    def mean(self) -> torch.Tensor:
        return self.loc

    @property
    def mode(self) -> torch.Tensor:
        return self.loc

    @property
    def variance(self) -> torch.Tensor:
        return self.covariance_vector

    def log_prob(self, value):
        logprob = - 0.5 * self.dim * torch.log(torch.tensor(2*torch.pi)) - 0.5 * self.covariance_vector.log().sum(-1) \
            - 0.5 * ((value - self.loc).square() / self.covariance_vector).sum(-1)
        return logprob

    def sample(self, sample_shape: torch.Size = torch.Size()):
        return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        s = self._normal.sample(sample_shape+self._shape)
        s = self.covariance_vector.sqrt() * s + self.loc
        return s

    def get_diagnostics(self):
        return OrderedDict({
            "mean": self.mean.detach().numpy(),
            "std": self.stddev.detach().numpy(),
        })
