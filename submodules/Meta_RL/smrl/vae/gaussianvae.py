"""
This module contains implementations of encoders, decoders and a VAE (see basevae.py)
with Gaussian prior, likelihood, and posteriors.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-22

Note:
    The code in here is not used for Reinforcement Learning. It is here
    for VAE purposes only (e.g. training a VAE on MNIST data).
    If you are interested in Meta Reinforcement Learning, you can ignore this
    file.
"""

from typing import Union, Tuple, List
import torch
from torch.distributions import Distribution
from ..utility.distributions import DiagonalMultivariateNormal

from .basevae import VAE, Encoder, Decoder

class GaussianEncoder(Encoder):
    """An encoder which returns the mean vector and vector of logarithmic diagonal entries of a diagonal covariance matrix.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 32, num_channels: int = None, hidden_layers: int = 2) -> None:
        super().__init__(input_dim, latent_dim)

        self.num_channels = num_channels
        if num_channels is None:
            num_channels = 1

        assert hidden_layers > 0, "Please choose at least one hidden layer!"
        layers = []
        layers.append(torch.nn.Linear(input_dim * num_channels, hidden_dim))
        layers.append(torch.nn.ReLU())
        for _ in range(hidden_layers-1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, 2*latent_dim))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Distribution:
        if self.num_channels is not None:
            x = x.reshape(*x.shape[:-2], x.shape[-1]*x.shape[-2])
        x = self.network(x)
        mu, log_sigma = x.split(self.latent_dim, dim=-1)
        covariance = log_sigma.exp()
        return DiagonalMultivariateNormal(loc=mu, covariance_vector=covariance)

class GaussianDecoder(Decoder):
    """A decoder which returns the mean vector and vector of logarithmic diagonal entries of a diagonal covariance matrix.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 32, num_channels: int = None, \
                 hidden_layers: int = 2, log_std: float = 0.0, train_std: bool = False) -> None:
        super().__init__(input_dim, latent_dim)

        self.num_channels = num_channels

        assert hidden_layers > 0, "Please choose at least one hidden layer!"
        layers = []
        layers.append(torch.nn.Linear(latent_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        for _ in range(hidden_layers-1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, 2*input_dim))

        self.network = torch.nn.Sequential(*layers)

        log_std = torch.ones([input_dim]) * log_std
        self.log_std = torch.nn.Parameter(log_std, requires_grad=train_std)

    def forward(self, z: torch.Tensor) -> Distribution:
        z = self.network(z)
        mu, log_sigma = z.split(self.input_dim, dim=-1)
        log_sigma = -5 * torch.ones_like(log_sigma).to(self.device) # OPTION: Remove this line to make log_sigma trainable
        if self.num_channels is not None:
            mu = mu.unsqueeze(-2)
            log_sigma = log_sigma.unsqueeze(-2)
        covariance = self.log_std.exp().square()
        return DiagonalMultivariateNormal(loc=mu, covariance_vector=covariance)

class GaussianConvEncoder(Encoder):
    """An encoder which returns the mean vector and vector of logarithmic diagonal entries of a diagonal covariance matrix.
    It is tailored for images of size (1,28,28) and uses Conv2d.
    """
    def __init__(self, input_dim: Tuple[int], latent_dim: int) -> None:
        super().__init__(input_dim, latent_dim)

        assert input_dim == 28*28, "Input dimension must be images of shape (28, 28)!"

        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=1), # (1,28,28) -> (8,14,14)
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=2),    # (8,14,14) -> (16,7,7)
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),   # (16,7,7) -> (32,3,3)
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Flatten(), # (32,3,3) -> (32*3*3)
            torch.nn.Linear(in_features=32*3*3, out_features=2*latent_dim)
        )

    def forward(self, x: torch.Tensor) -> Distribution:
        x = x.view(*(x.shape[:-1]), 28, 28)
        x = self.network(x)
        mean, logvar = x.split(self.latent_dim, -1)
        covariance = logvar.exp()
        return DiagonalMultivariateNormal(loc=mean, covariance_vector=covariance)

class GaussianConvDecoder(Decoder):
    """A decoder which returns the mean vector and vector of logarithmic diagonal entries of a diagonal covariance matrix.
    It is tailored for images of size (28,28) and uses ConvTranspose2d.
    """

    def __init__(self, input_dim: int, latent_dim: int, train_std: bool = False, log_std: float = 0.0) -> None:
        super().__init__(input_dim, latent_dim)

        assert input_dim == 28*28, "Input dimension must be images of shape (28, 28)!"
        self.input_dim = input_dim

        self.network = GaussianConvDecoder.ConvTransposeNetwork(input_dimension=latent_dim, output_dimension=input_dim)

        log_std = torch.ones([input_dim]) * log_std
        self.log_std = torch.nn.Parameter(log_std, requires_grad=train_std)

    class ConvTransposeNetwork(torch.nn.Module):
        def __init__(self, input_dimension: int, output_dimension: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(in_features=input_dimension, out_features=8*3*3)  # (2) -> (8*3*3)
            self.conv_transpose = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),   # (8,3,3) -> (16,7,7)
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1), # (16,7,7) -> (16,14,14)
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1), # (16,14,14) -> (1,28,28)
            )
            self.input_dimension = input_dimension
            self.output_dimension = output_dimension

        def forward(self, x):
            batch_size = x.shape[:-1]
            x: torch.Tensor = self.linear(x)
            bs = 1
            for d in batch_size:
                bs *= d
            x = x.view(bs, 8, 3, 3)    # Necessary because ConvTranspose2d can only work with 4d tensors
            x = self.conv_transpose(x)
            return x.view(*batch_size, 1, self.output_dimension)   # Undo reshaping to obtain 5d tensor (or 4d)

        def train(self, mode: bool = True) -> torch.nn.Module:
            if mode:
                self.linear.train()
                self.conv_transpose.train()
                return self
            else:
                return self.eval()

        def eval(self) -> torch.nn.Module:
            self.linear.eval()
            self.conv_transpose.eval()
            return self

        def parameters(self):
            return list(self.linear.parameters()) + list(self.conv_transpose.parameters())

        def to(self, device):
            self.linear.to(device)
            self.conv_transpose.to(device)

    def forward(self, z: torch.Tensor) -> Distribution:
        mean = self.network(z) # Shape (*batch_size, 2, 28*28)
        covariance = self.log_std.exp().square()
        return DiagonalMultivariateNormal(mean, covariance)

class GaussianVAE(VAE):
    """A variational autoencoder which uses Gaussian distributions with diagonal covariance matrices
    for the latent prior and posterior as well as for the likelihood distribution.

    Parameters
    ----------
    input_dim : int
        Dimension of the original space (inputs).
    latent_dim = int
        Dimension of the latent space.
    beta : float, optional
        Parameter beta of a beta-VAE. See VAE for more details and references, by default 1.0
    hidden_layers : int, optional
        Number of hidden layers in encoder and decoder, by default 1
    """
    def __init__(self, input_dim: int, latent_dim: int, num_channels: int = None, beta: float = 1.0, hidden_layers: int = 1) -> None:
        # encoder = GaussianEncoder(input_dim, latent_dim, hidden_layers=hidden_layers, num_channels=num_channels)
        # decoder = GaussianDecoder(input_dim, latent_dim, hidden_layers=hidden_layers, num_channels=num_channels, log_std=0.0, train_std=True)
        encoder = GaussianConvEncoder(input_dim=28*28, latent_dim=2)
        decoder = GaussianConvDecoder(input_dim=28*28, latent_dim=2, log_std=-2.5, train_std=False)
        super().__init__(encoder, decoder, beta)
        prior_mean = torch.zeros(latent_dim)
        prior_cov = torch.ones(latent_dim)
        self.prior = DiagonalMultivariateNormal(prior_mean, covariance_vector=prior_cov)

    def kl_divergence(self, posterior: DiagonalMultivariateNormal) -> torch.Tensor:
        """Computes the exact KL divergence of the latent encoded posterior and the latent prior. 
        Since both distributions are Gaussians, an analytical expression for the KL divergence is known:

        Let q ~ N(m1, S1), p ~ N(m2, S2) be two Gaussian distributions of dimension d
        with means m1, m2 and covariance matrices S1, S2, respectively. Then

        KL(q||p) = 0.5 * ( tr{ Sp^(-1) * Sq } + (mp-mq)^T * Sp^(-1) * (mp-mq) - d + ln(|Sp|) - ln(|Sq|) )

        Parameters
        ----------
        latent_dist_params : Tuple[torch.Tensor]
            Distribution parameters from the encoder, based on input variables.
            For the Gaussian VAE, the tuple contains the mean and logarithmic covariance vector.
            Shape (batch_size, dist_param_dim)

        Returns
        -------
        torch.Tensor
            KL divergence between posterior and prior, Shape (batch_size)
        """
        mu: torch.Tensor = posterior.mean
        sigma: torch.Tensor = posterior.variance.diag()
        kl_div = 0.5 * (
            (sigma / self.prior.variance).sum(-1)  # Trace
            + ((self.prior.mean - mu).square() / self.prior.variance).sum(-1)
            - self.latent_dim
            + self.prior.variance.log().sum() - sigma.log().sum(-1)
        )
        return kl_div

    def elbo(self, x: torch.Tensor, n_latent_samples: int) -> torch.Tensor:
        # Sample latent representations from approximate posterior
        latent_dist = self.encoder.forward(x)
        z = latent_dist.rsample(sample_shape=torch.Size([n_latent_samples])) # Shape (n_latent_samples, batch_size, latent_dim)
        out_dist = self.decoder.forward(z)

        # Get logarithmic likelihood and KL divergence
        kl_divergence = self.kl_divergence(latent_dist)  # Shape (batch_size)
        loglikelihood = out_dist.log_prob(x)      # Shape (n_latent_samples, batch_size)

        # print(f"Reconstruction quality: {loglikelihood.mean().item()} \t KL divergence: {kl_divergence.mean().item()}")

        # Compute beta-ELBO
        elbo: torch.Tensor = loglikelihood - self.beta * kl_divergence.unsqueeze(0)
        elbo = elbo.mean(dim = 0)

        return elbo

    def to(self, device):
        prior_mean = torch.zeros(self.latent_dim).to(device)
        prior_cov = torch.ones(self.latent_dim).to(device)
        self.prior = DiagonalMultivariateNormal(prior_mean, covariance_vector=prior_cov)
        super().to(device)