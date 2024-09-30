"""
This module contains the base classes Encoder, Decoder, and VAE
which can be used for implementing a structured Variational Autoencoder.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2022-11-14
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, List
import os
from os import path
import torch
import numpy as np
from torch.distributions.distribution import Distribution
from torch.utils.data import DataLoader
from tqdm import tqdm

class Encoder(torch.nn.Module, ABC):
    """Base class for encoders.

    Parameters
    ----------
    input_dim : int
        Input dimension
    latent_dim : int
        Latent dimension
    """
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self._device = torch.device('cpu')

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Distribution:
        """Returns the parameters for the posterior probability q(z|x) based on the input x.
        The distribution type is specified within the VAE main class.

        Parameters
        ----------
        x : torch.Tensor
            Input variable

        Returns
        -------
        Distribution
            Distribution over latent space based on input, q(z|x)
        """
        raise NotImplementedError

    def to(self, device: torch.device):
        super().to(device)
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

class Decoder(torch.nn.Module, ABC):
    """Base class for decoders.

    Parameters
    ----------
    input_dim : int
        Input dimension
    latent_dim : int
        Latent dimension
    """
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self._device = torch.device('cpu')

    @abstractmethod
    def forward(self, z: torch.Tensor) -> Distribution:
        """Returns the parameters for the likelihood p(x|z) based on the latent encoding z.
        The distribution type is specified within the VAE main class.

        Parameters
        ----------
        z : torch.Tensor
            Latent encoding

        Returns
        -------
        Distribution
            Distribution over predictions based on latent samples, p(x|z)
        """
        raise NotImplementedError

    def to(self, device: torch.device):
        super().to(device)
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

class VAE(ABC):
    """Base class for Variational autoencoders.

    Parameters
    ----------
    encoder : Encoder
        Encoder network
    decoder : Decoder
        Decoder network
    beta : float, optional
        Hyperparameter beta used for balancing approximation accuracy and prior-following.
        Refer to Higgins et al. (2017), "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" for more information.
        By default 1.0

    References
    ----------
    * Kingma & Welling (2020), "Auto-Encoding Variational Bayes"
    * Kingma & Welling (2019), "An Introduction to Variational Autoencoders"
    * Higgins et al. (2017), "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, prior: Distribution = None, beta: float = 1.0) -> None:
        self.encoder = encoder
        self.decoder = decoder
        assert self.encoder.input_dim == self.decoder.input_dim, "Encoder and decoder must share the same input/original dimension!"
        assert self.encoder.latent_dim == self.decoder.latent_dim, "Encoder and decoder must share the same latent dimension!"
        self.input_dim = encoder.input_dim
        self.latent_dim = encoder.latent_dim
        self.beta = beta        # Hyperparameter beta as in Beta-VAE, see Higgins et al. (2017), "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
        self.prior: Distribution = prior       # Prior distribution, for regularization
        self._device: torch.device = torch.device('cpu')

    @property
    def device(self) -> torch.device:
        return self._device

    def elbo(self, x: torch.Tensor, n_latent_samples: int) -> torch.Tensor:
        """Returns an estimate of the Evidence Lower Bound (ELBO) based on Monte Carlo estimation.

        ELBO = E_{z ~ q(z|x)} [log p(x|z) + beta * (log p(z) - log q(z|x))]
             = E_{z ~ q(z|x)} [log p(x|z)] - beta * KL( q(z|x) || p(z) )

        Parameters
        ----------
        x : torch.Tensor
            Input samples used for ELBO estimation.
            Shape (batch_size, input_dim)
        n_latent_samples: int
            Number of latent samples (from the approximate posterior) for Monte Carlo approximation.

        Returns
        -------
        torch.Tensor
            The ELBO for each of the instances in x.
            Shape (batch_size)
        """
        # Sample latent representations
        latent_dist = self.encoder.forward(x)
        z = latent_dist.rsample(sample_shape=torch.Size([n_latent_samples])) # Shape (latent_samples, batch_size, latent_dim)
        output_dist = self.decoder.forward(z)

        # Get logarithmic probabilities
        logprior = self.prior.log_prob(z)         # Shape (n_latent_samples, batch_size)
        logposterior = latent_dist.log_prob(z)    # Shape (n_latent_samples, batch_size)
        loglikelihood = output_dist.log_prob(x)     # Shape (n_latent_samples, batch_size)

        # Compute beta-ELBO
        elbo: torch.Tensor = loglikelihood + self.beta * (logprior - logposterior)
        elbo = elbo.mean(dim = 0)

        return elbo

    def parameters(self) -> List[torch.nn.Parameter]:
        """Returns the parameters of the encoder and decoder network.

        Returns
        -------
        List[torch.nn.Parameter]
            Encoder and decoder network parameters
        """
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def to(self, device):
        self._device = device
        self.encoder.to(device)
        self.decoder.to(device)
        if self.prior is not None and self.prior.sample(torch.Size([1])).device != self.device:
            Warning("The prior is not on the same device as the encoder and decoder networks! "
                    + "Please move the prior to device manually.")

    def save_networks(self, filepath: str):
        """Save encoder and decoder networks to a file.

        Parameters
        ----------
        path : str
            Path to file
        """
        os.makedirs(path.dirname(filepath), exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict()
        }, filepath)

    def load_networks(self, filepath: str):
        """Load encoder and decoder network from a file.

        Parameters
        ----------
        path : str
            Path to file
        """
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])


def train_vae(vae: VAE, training_data: DataLoader, epochs: int, latent_samples: int = 100,
    optimizer_type: torch.optim.Optimizer = torch.optim.Adam, lr: float = 1e-3,
    optimizer_params: dict = None, save_checkpoints_to: str = None, 
    encoder_train_ratio: int = 1) -> List[float]:
    """Train a Variational Autoencoder based on training data.

    Parameters
    ----------
    vae : VAE
        The model to train.
    training_data : DataLoader
        The training data (DataLoader).
    epochs : int
        Number of epochs used for training.
        Note that the length of an epoch depends on the DataLoader-object training_data!
    latent_samples : int, optional
        The number of latent samples used for Monte Carlo approximation of the ELBO, by default 100
    optimizer_type : torch.optim.Optimizer, optional
        The optimizer to use (e.g. SGD, Adam), by default torch.optim.Adam
    lr : float, optional
        The learning rate, by default 1e-3
    optimizer_params : dict, optional
        Additional parameters which should be passed to the optimizer, by default None
    save_checkpoints_to: str, optional
        Path to directory where checkpoints can be stored, by default None
    encoder_train_ratio: int, optional
        Focuses on encoder training more aggressively by repeating optimization more often, by default 1
    Returns
    -------
    List[float]
        List of training iteration losses.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type}")

    vae.to(device)

    losses = []
    if optimizer_params is None: optimizer_params = {}
    optimizer_encoder: torch.optim.Optimizer = optimizer_type(lr=lr, params=vae.encoder.parameters(), **optimizer_params)
    optimizer_decoder: torch.optim.Optimizer = optimizer_type(lr=lr, params=vae.decoder.parameters(), **optimizer_params)

    vae.encoder.train()
    vae.decoder.train()

    for epoch in range(epochs):
        data_iterator = tqdm(training_data, desc=f"Epoch {epoch+1} of {epochs}")
        for i, x in enumerate(data_iterator):
            x = x[0]    # Ignore labels, etc
            x = x.to(device)
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss = -vae.elbo(x, latent_samples).mean()
            loss.backward()
            optimizer_encoder.step()
            decoder_update = i % encoder_train_ratio == 0
            if decoder_update:
                optimizer_decoder.step()
            losses.append(loss.detach().item())
            data_iterator.set_postfix({"Loss": np.mean(losses[-50:]), "Decoder update": decoder_update})

        if save_checkpoints_to is not None:
            filepath = path.join(save_checkpoints_to, f'epoch_{epoch+1:04d}')
            vae.save_networks(filepath)

    vae.encoder.eval()
    vae.decoder.eval()

    return losses
