""" 
This module contains base classes for Encoders, Decoders, and VAEs 
which are specifically taylored for MDPs.

In particular, the Encoders must process information from trajectories,
i.e. they have inputs (state, actions, rewards, next_states).
Decoders must provide reconstructions of rewards and next states.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-03-04
"""

from abc import ABC, abstractmethod
import torch
from torch.distributions.distribution import Distribution
import numpy as np
from typing import List, Tuple, Union, Dict

from rlkit.torch.core import torch_ify, np_ify

from .basevae import Encoder, Decoder, VAE
from ..utility.distributions import DiagonalMultivariateNormal
from ..utility import console_strings as console_strings

class MdpEncoder(Encoder, ABC):
    """An encoder for MDPs which takes as inputs
    - observations
    - actions
    - rewards
    - next observations
    and outputs the (approximate) latent posterior q(z|x).
    
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
    context_size : int
        Length of the context sequences used for encoding.
    encoding_mode : str, optional
        Determines how encodings from ``get_encoding()`` and ``get_encodings()``
        are generated from the latent posterior. Accepted values are: \n
        | ``'sample'`` | ``'mean'`` | ``'mode'`` | ``'mean_var'`` | \n
        See the documentation of these two functions for detailed information on
        the possible values.
        You can change the value later by setting the property ``encoding_mode``
        to any of the values above.
        By default 'sample'
    """
    supports_variable_sequence_length = False   # Whether this encoder can handle sequences of variable length
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        latent_dim: int,
        encoding_dim: int,
        context_size: int = None,
        encoding_mode: str = 'sample',
        *args,
        **kwargs,
    ) -> None:
        super().__init__(input_dim=-1, latent_dim=latent_dim)   # Note that input_dim doesn't make sense for MdpEncoder
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.context_size = context_size
        self.encoding_mode = encoding_mode
        self.encoding_dim = encoding_dim

    @abstractmethod
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        terminals: torch.Tensor,
    ) -> torch.distributions.Distribution:
        """Returns the posterior probability q(z|x) based on MDP context x.

        Parameters
        ----------
        observations : torch.Tensor
            Context of observations, shape (batch_size, context_size, obs_dim)
        actions : torch.Tensor
            Context of actions, shape (batch_size, context_size, action_dim)
        rewards : torch.Tensor
            Context of rewards, shape (batch_size, context_size, 1)
        next_observations : torch.Tensor
            Context of next observations, shape (batch_size, context_size, obs_dim)
        terminals : torch.Tensor
            Context of terminal indicators (~ end of trajectory),
            shape (batch_size, context_size, 1)

        Returns
        -------
        torch.distributions.Distribution
            Distribution over the latent space q(z|x).
            Distribution parameters have shape (batch_size, latent_dim)
        """
        raise NotImplementedError

    def get_encoding(
        self,
        observations: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        rewards: Union[np.ndarray, torch.Tensor],
        next_observations: Union[np.ndarray, torch.Tensor],
        terminals: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Returns a sample from the latent space based on context
        data. NOT BATCHED! (See `get_encodings()` for batched version.)

        The encodings are generated from the latent posterior based on the input
        property ``self.encoding_mode``:
        - ``'sample'``: Sample a variable from the latent distribution (random)
        - ``'mean'``: Take the mean of the latent distribution (deterministic)
        - ``'mode'``: Take the mode of the latent distribution (deterministic) 
                (May not be supported by all distribution types)
        - ``'mean_var'``: Concatenate mean and variance (deterministic, with uncertainty information)

        Parameters
        ----------
        observations : Union[np.ndarray, torch.Tensor]
            Context of observations, shape (context_size, obs_dim)
        actions : Union[np.ndarray, torch.Tensor]
            Context of actions, shape (context_size, action_dim)
        rewards : Union[np.ndarray, torch.Tensor]
            Context of rewards, shape (context_size, 1)
        next_observations : Union[np.ndarray, torch.Tensor]
            Context of next observations, shape (context_size, obs_dim)
        terminals : Union[np.ndarray, torch.Tensor]
            Context of terminal indicators (~ end of trajectory),
            shape (context_size, 1)

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Sample z ~ q(z|x),
            Same type as input data.
        """
        encoding = self.get_encodings(
            observations[None],
            actions[None],
            rewards[None],
            next_observations[None],
            terminals[None],
        )
        return encoding.squeeze(axis=0)

    def get_encodings(
        self,
        observations: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        rewards: Union[np.ndarray, torch.Tensor],
        next_observations: Union[np.ndarray, torch.Tensor],
        terminals: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Returns an encoding based on context data, batched.
        
        The encodings are generated from the latent posterior based on the input
        property ``self.encoding_mode``:
        - ``'sample'``: Sample a variable from the latent distribution (random)
        - ``'mean'``: Take the mean of the latent distribution (deterministic)
        - ``'mode'``: Take the mode of the latent distribution (deterministic) 
                (May not be supported by all distribution types)
        - ``'mean_var'``: Concatenate mean and variance (deterministic, with uncertainty information)

        Parameters
        ----------
        observations : Union[np.ndarray, torch.Tensor]
            Context of observations, shape (batch_size, context_size, obs_dim)
        actions : Union[np.ndarray, torch.Tensor]
            Context of actions, shape (batch_size, context_size, action_dim)
        rewards : Union[np.ndarray, torch.Tensor]
            Context of rewards, shape (batch_size, context_size, 1)
        next_observations : Union[np.ndarray, torch.Tensor]
            Context of next observations, shape (batch_size, context_size, obs_dim)
        terminals : Union[np.ndarray, torch.Tensor]
            Context of terminal indicators (~ end of trajectory),
            shape (batch_size, context_size, 1)

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Sample z ~ q(z|x),
            Same type as input data.
        """
        dist = self.forward(
            torch_ify(observations),
            torch_ify(actions),
            torch_ify(rewards),
            torch_ify(next_observations),
            torch_ify(terminals),
        )

        if self.encoding_mode == 'sample':
            encoding = dist.rsample()
        elif self.encoding_mode == 'mean':
            encoding = dist.mean
        elif self.encoding_mode == 'mode':
            encoding = dist.mode    # May not be supported for all distribution types.
        elif self.encoding_mode == 'mean_var':
            encoding = torch.cat(
                (
                    dist.mean,
                    torch.reshape(dist.variance, list(dist.mean.shape[:-1]) + [-1])
                ),
                dim=-1
            )
        else:
            raise ValueError(f"Value {self.encoding_mode} for property ``self.encoding_mode`` is not supported!")
        
        if isinstance(observations, np.ndarray):
            return np_ify(encoding)
        else:
            return encoding
        

class MdpDecoder(Decoder, ABC):
    """A decoder for MDPs which takes as inputs

    - latent samples z
    - the latest observation o
    - the latest action a

    and outputs distributions for

    - the reward predictions r
    - next observation predictions o'

    Parameters
    ----------
    latent_dim : int
        Dimension of latent representations.
    observation_dim : int
        Dimension of observations in the MDP.
    action_dim : int
        Dimension of the action space in the MDP.
    """
    def __init__(
        self,
        latent_dim: int,
        observation_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__(input_dim=-1, latent_dim=latent_dim)  # Note that input_dim doesn't make sense for MdpDecoder
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    @abstractmethod
    def forward(self, z: torch.Tensor, observation: torch.Tensor, action: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.distributions.Distribution]:
        """Returns the the likelihoods p(reward|z, observation, action) 
        and p(next observation|z, observation, action) based on the latent encoding z,
        observations, and actions.

        Parameters
        ----------
        z : torch.Tensor
            Latent encoding
        observation : torch.Tensor
            Observations, shape (batch_size, target_size, obs_dim)
        action : torch.Tensor
            Actions, shape (batch_size, target_size, action_dim)

        Note: batch_size can have multiple dimensions.

        Returns
        -------
        Tuple[torch.distributions.Distribution, torch.distributions.Distribution]
            Distribution over the rewards p(reward|z, observation, action)
            Distribution over the next observations p(next observation|z, observation, action),
            Shape of the parameters is: (batch_size, target_size, <1 or obs_dim>)
        """
        raise NotImplementedError

class MdpVAE(VAE, ABC):
    """Base class for MDP-VAEs. Implements ``elbo()``.

    MDP-VAEs accept contexts (e.g. a transition history) which they
    encode (see `MdpEncoder`) to a latent representation `z`.
    The latent representation, together with a current state-action pair,
    is used to predict the reward and next state (see `MdpDecoder`).

    Parameters
    ----------
    encoder : Encoder
        Encoder network
    decoder : Decoder
        Decoder network
    prior : Distribution, optional
        Prior distribution (for regularization / KL divergence),
        by default a standard normal distribution (using ``DiagonalMultivariateNormal``)
    beta : float, optional
        Hyperparameter beta used for balancing approximation accuracy and prior-following.
        Refer to Higgins et al. (2017), "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" for more information.
        By default 1.0

    References
    ----------
    Basic VAEs:
    * Kingma & Welling (2020), "Auto-Encoding Variational Bayes"
    * Kingma & Welling (2019), "An Introduction to Variational Autoencoders"
    * Higgins et al. (2017), "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"

    Meta-learning VAEs for MDPs:
    * Rakelly et al. (2019), "Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic  Context Variables"
    
    Neural Processes:
    * Garnelo et al. (2018), "Neural Processes" in ICML Workshop on Theoretical Foundations and Applications of Deep Generative Models
    * Garnelo et al. (2018), "Conditional Neural Processes", ICML 2018

    Implementation details
    ----------------------
    * elbo: Implementation-specific version of the Expectation Lower BOund
    """
    def __init__(self, encoder: MdpEncoder, decoder: MdpDecoder, prior: Distribution = None, beta: float = 1.0) -> None:
        self.encoder: MdpEncoder
        self.decoder: MdpDecoder
        self._default_prior = False
        if prior is None:
            self._default_prior = True
            prior = DiagonalMultivariateNormal(torch.zeros((encoder.latent_dim)), torch.ones((encoder.latent_dim)))
        super().__init__(encoder, decoder, prior, beta)

    def elbo(
        self,
        context: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        n_latent_samples: int = 100,
    ) -> torch.Tensor:
        """Returns a Monte-Carlo estimate of the Expectation Lower BOund (ELBO).

        In the MDP setting, the encoder is provided with a context (past transitions)
        which it maps to a latent distribution. From this latent distribution, latent samples
        are drawn (Monte Carlo estimation). The observation and action can be used to
        predict the reward and next observation. This is achieved by the decoder, which
        also takes into account the latent sample.

        Parameters
        ----------
        context : Dict[str, torch.Tensor]
            A context dictionary with entries:
                "observations", shape (batch_size, context_size, obs_dim)
                "actions", shape (batch_size, context_size, action_dim)
                "rewards", shape (batch_size, context_size, 1)
                "next_observations", shape (batch_size, context_size, obs_dim)
                "terminals", shape (batch_size, context_size, 1)
        target : Dict[str, torch.Tensor]
            A prediction target dictionary with entries:
                "observations", shape (batch_size, target_size, obs_dim)
                "actions", shape (batch_size, target_size, action_dim)
                "rewards", shape (batch_size, target_size, 1)
                "next_observations", shape (batch_size, target_size, obs_dim)
                "terminals", shape (batch_size, target_size, 1)
        n_latent_samples : int, optional
            Number of latent samples to use for Monte Carlo estimation,
            by default 100

        Returns
        -------
        torch.Tensor
            ELBO, shape (batch_size)
        """
        batch_size = context['observations'].shape[0]
        target_size = target['observations'].shape[1]
        # Encode context and sample latent variables
        latent_dist = self.encoder.forward(
            observations=context["observations"],
            actions=context["actions"],
            rewards=context["rewards"],
            next_observations=context["next_observations"],
            terminals=context["terminals"],
        )
        z = latent_dist.rsample(sample_shape=torch.Size([n_latent_samples]))    # shape (n_samples, batch_size, latent_dim)
        z_ = z.unsqueeze(2).repeat(1, 1, target_size, 1)    # shape (n_samples, batch_size, target_size, latent_dim)

        # Decode context + observation + action to predict reward and next observation
        reward_dist, obs_dist = self.decoder.forward(z_, target['observations'], target['actions'])
        
        # Likelihoods of provided rewards and next observation under the predicted distributions
        reward_loglikelihood = reward_dist.log_prob(target['rewards']).sum(-1)
        obs_loglikelihood = obs_dist.log_prob(target['next_observations']).sum(-1)

        # Posterior probability of latent samples under provided context
        z_logposterior = latent_dist.log_prob(z)
        # Prior probability of latent samples
        z_logprior = self.prior.log_prob(z)

        # Accumulate ELBO
        elbo = reward_loglikelihood + obs_loglikelihood - self.beta * (z_logposterior - z_logprior)
        elbo = elbo.mean(dim=0)

        return elbo

    def train(self, mode=True):
        """Sets the VAE in training mode. 
        
        This has only effects on certain encoders or decoders, e.g. if they use dropout.

        This function ensures that the VAE can be used similar to a torch.nn.Module.

        Parameters
        ----------
        mode : bool, optional
            training mode (True) or test mode (False), by default True
        """
        self.encoder.train(mode)
        self.decoder.train(mode)

    def to(self, device):
        if self._default_prior:
            self.prior = DiagonalMultivariateNormal(torch.zeros((self.encoder.latent_dim), device=device), torch.ones((self.encoder.latent_dim), device=device))
        super().to(device)


class MdpIWAE(MdpVAE):
    """Importance-weighted autoencoder for MDP prediction.

    MDP-VAEs accept contexts (e.g. a transition history) which they
    encode (see `MdpEncoder`) to a latent representation `z`.
    The latent representation, together with a current state-action pair,
    is used to predict the reward and next state (see `MdpDecoder`).

    Parameters
    ----------
    encoder : Encoder
        Encoder network
    decoder : Decoder
        Decoder network
    prior : Distribution, optional
        Prior distribution (for regularization / KL divergence),
        by default a standard normal distribution (using ``DiagonalMultivariateNormal``)
    beta : float, optional
        Hyperparameter beta used for balancing approximation accuracy and prior-following.
        Refer to Higgins et al. (2017), "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" for more information.
        By default 1.0

    References
    ----------
    Basic VAEs:
    * Kingma & Welling (2020), "Auto-Encoding Variational Bayes"
    * Kingma & Welling (2019), "An Introduction to Variational Autoencoders"
    * Higgins et al. (2017), "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
    * Burda et al. (2015), "Importance Weighted Autoencoders"

    Meta-learning VAEs for MDPs:
    * Rakelly et al. (2019), "Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic  Context Variables"
    
    Neural Processes:
    * Garnelo et al. (2018), "Neural Processes" in ICML Workshop on Theoretical Foundations and Applications of Deep Generative Models
    * Garnelo et al. (2018), "Conditional Neural Processes", ICML 2018

    Implementation details
    ----------------------
    * elbo: Implementation-specific version of the Expectation Lower BOund
    """
    def elbo(
        self,
        context: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        n_latent_samples: int = 100,
    ) -> torch.Tensor:
        batch_size = context['observations'].shape[0]
        target_size = target['observations'].shape[1]
        # Encode context and sample latent variables
        latent_dist = self.encoder.forward(
            observations=context["observations"],
            actions=context["actions"],
            rewards=context["rewards"],
            next_observations=context["next_observations"],
            terminals=context["terminals"],
        )
        z = latent_dist.rsample(sample_shape=torch.Size([n_latent_samples]))    # shape (n_samples, batch_size, latent_dim)
        z_ = z.unsqueeze(2).repeat(1, 1, target_size, 1)    # shape (n_samples, batch_size, target_size, latent_dim)

        # Decode context + observation + action to predict reward and next observation
        reward_dist, obs_dist = self.decoder.forward(z_, target['observations'], target['actions'])
        
        # Likelihoods of provided rewards and next observation under the predicted distributions
        reward_loglikelihood = reward_dist.log_prob(target['rewards']).sum(-1)
        obs_loglikelihood = obs_dist.log_prob(target['next_observations']).sum(-1)

        # Posterior probability of latent samples under provided context
        z_logposterior = latent_dist.log_prob(z)
        # Prior probability of latent samples
        z_logprior = self.prior.log_prob(z)

        # Accumulate ELBO
        elbo = torch.logsumexp(
                reward_loglikelihood + obs_loglikelihood \
                    - self.beta * (z_logposterior - z_logprior),
                dim = 0
            ) - torch.log(torch.tensor(1 / n_latent_samples, device=self.device))
        return elbo


class NeuralProcess(MdpVAE):
    """Neural Processes, a modified version of ``MdpVAE`` which defines a different
    ELBO.

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
    Neural Processes:
    * Garnelo et al. (2018), "Neural Processes" in ICML Workshop on Theoretical Foundations and Applications of Deep Generative Models
    * Garnelo et al. (2018), "Conditional Neural Processes", ICML 2018
    """
    def __init__(self, encoder: MdpEncoder, decoder: MdpDecoder, beta: float = 1) -> None:
        super().__init__(encoder, decoder, None, beta)
        if not encoder.supports_variable_sequence_length:
            print(console_strings.warning("WARNING: The encoder does not support variable sequence lengths! This might lead to undesirable behavior with Neural Process inference!"))

    def elbo(
        self,
        context: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        n_latent_samples: int = 100,
    ) -> torch.Tensor:
        batch_size = context['observations'].shape[0]
        target_size = target['observations'].shape[1]

        # Extended context, used for sampling z
        context_and_target = dict(
            observations=torch.cat((context["observations"], target["observations"]), dim=1),
            actions=torch.cat((context["actions"], target['actions']), dim=1),
            rewards=torch.cat((context["rewards"], target['rewards']), dim=1),
            next_observations=torch.cat((context["next_observations"], target['next_observations']), dim=1),
            terminals=torch.cat((context["terminals"], target['terminals']), dim=1),
        )
        # Shuffle to make encoding by ordering impossible (otherwise the encoder could just encode the targets)
        idx = torch.randperm(context_and_target['observations'].shape[1])
        for key, tensor_value in context_and_target.items():
            # Shuffle along time dimension such that batches and environment dimensions 
            # are still ordered and transition information is still grouped together!
            context_and_target[key] =  tensor_value[:, idx, :]  
        
        # Encode context AND target, sample latent variables
        latent_dist = self.encoder.forward(**context_and_target)
        z = latent_dist.rsample(sample_shape=torch.Size([n_latent_samples]))    # shape (n_samples, batch_size, latent_dim)
        z_ = z.unsqueeze(2).repeat(1, 1, target_size, 1)    # shape (n_samples, batch_size, target_size, latent_dim)

        # Decode context + observation + action to predict reward and next observation
        reward_dist, obs_dist = self.decoder.forward(z_, target['observations'], target['actions'])
        
        # Likelihoods of provided rewards and next observation under the predicted distributions
        reward_loglikelihood = reward_dist.log_prob(target['rewards']).sum(-1)
        obs_loglikelihood = obs_dist.log_prob(target['next_observations']).sum(-1)

        # Posterior probability of latent samples under provided context
        z_logposterior = latent_dist.log_prob(z)

        # Posterior probability of latent samples under context (without target) -> 'reduced'
        reduced_latent_dist = self.encoder.forward(**context)
        z_logposterior_reduced = reduced_latent_dist.log_prob(z)

        # Accumulate ELBO
        elbo = reward_loglikelihood + obs_loglikelihood + self.beta * (z_logposterior_reduced - z_logposterior)
        elbo = elbo.mean(dim=0) # Average over Monte-Carlo samples

        return elbo

class IWNeuralProcess(NeuralProcess):
    """An importance-weighted Neural Processes. This neural process combines 
    the idea of neural processes with the tighter lower bound from Importance
    Weighted Autoencoders (IWAE). 

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
    Neural Processes:
    * Garnelo et al. (2018), "Neural Processes" in ICML Workshop on Theoretical Foundations and Applications of Deep Generative Models
    * Garnelo et al. (2018), "Conditional Neural Processes", ICML 2018
    * Burda et al. (2015), "Importance Weighted Autoencoders"
    """
    def elbo(
        self,
        context: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        n_latent_samples: int = 100,
    ) -> torch.Tensor:
        batch_size = context['observations'].shape[0]
        target_size = target['observations'].shape[1]

        # Extended context, used for sampling z
        context_and_target = dict(
            observations=torch.cat((context["observations"], target["observations"]), dim=1),
            actions=torch.cat((context["actions"], target['actions']), dim=1),
            rewards=torch.cat((context["rewards"], target['rewards']), dim=1),
            next_observations=torch.cat((context["next_observations"], target['next_observations']), dim=1),
            terminals=torch.cat((context["terminals"], target['terminals']), dim=1),
        )
        # Shuffle to make encoding by ordering impossible (otherwise the encoder could just encode the targets)
        idx = torch.randperm(context_and_target['observations'].shape[1])
        for key, tensor_value in context_and_target.items():
            # Shuffle along time dimension such that batches and environment dimensions 
            # are still ordered and transition information is still grouped together!
            context_and_target[key] =  tensor_value[:, idx, :]  
        
        # Encode context AND target, sample latent variables
        latent_dist = self.encoder.forward(**context_and_target)
        z = latent_dist.rsample(sample_shape=torch.Size([n_latent_samples]))    # shape (n_samples, batch_size, latent_dim)
        z_ = z.unsqueeze(2).repeat(1, 1, target_size, 1)    # shape (n_samples, batch_size, target_size, latent_dim)

        # Decode context + observation + action to predict reward and next observation
        reward_dist, obs_dist = self.decoder.forward(z_, target['observations'], target['actions'])
        
        # Likelihoods of provided rewards and next observation under the predicted distributions
        reward_loglikelihood = reward_dist.log_prob(target['rewards']).sum(-1)
        obs_loglikelihood = obs_dist.log_prob(target['next_observations']).sum(-1)

        # Posterior probability of latent samples under provided context
        z_logposterior = latent_dist.log_prob(z)

        # Posterior probability of latent samples under context (without target) -> 'reduced'
        reduced_latent_dist = self.encoder.forward(**context)
        z_logposterior_reduced = reduced_latent_dist.log_prob(z)

        elbo = torch.logsumexp(
                reward_loglikelihood + obs_loglikelihood \
                    + self.beta * (z_logposterior_reduced - z_logposterior),
                dim = 0
            ) - torch.log(torch.tensor(1 / n_latent_samples, device=self.device))
        return elbo

class InfoMaxMdpVAE(MdpVAE):
    def __init__(self, encoder: MdpEncoder, decoder: MdpDecoder, prior: Distribution = None, beta: float = 1.0, alpha: float = 1.0) -> None:
        super().__init__(encoder, decoder, prior, beta)
        self._alpha = alpha
        t_input_dim = (2 * self.encoder.observation_dim + self.encoder.action_dim + 2 ) * self.encoder.context_size + self.encoder.latent_dim
        self._t_network = torch.nn.Sequential(
            torch.nn.Linear(t_input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        self._t_optimizer = torch.optim.Adam(self._t_network.parameters(), lr=1e-3)
        print(console_strings.red("This MdpVAE type is deprecated since it has not been tested!"))

    def elbo(
        self,
        context: Dict[str, torch.Tensor],
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        terminal: torch.Tensor,
        n_latent_samples: int = 100,
    ) -> torch.Tensor:
        # Encode context and sample latent variables
        latent_dist = self.encoder.forward(
            observations=context["observations"],
            actions=context["actions"],
            rewards=context["rewards"],
            next_observations=context["next_observations"],
            terminals=context["terminals"],
        )
        z = latent_dist.rsample(sample_shape=torch.Size([n_latent_samples]))
        z_permuted = z[torch.randperm(z.shape[0])]

        # Decode context + observation + action to predict reward and next observation
        reward_dist, obs_dist = self.decoder.forward(z, observation, action)
        
        # Likelihoods of provided rewards and next observation under the predicted distributions
        reward_loglikelihood = reward_dist.log_prob(reward)
        obs_loglikelihood = obs_dist.log_prob(next_observation)

        # Posterior probability of latent samples under provided context
        z_logposterior = latent_dist.log_prob(z)
        # Prior probability of latent samples
        z_logprior = self.prior.log_prob(z)

        # t-function update 
        x = torch.cat((context['observations'], context['actions'], context['rewards'], context['next_observations'], context['terminals']), dim=-1)
        x = x.reshape((x.shape[0], -1)) # Stack sequence data to have shape (batch_dim, *)
        x = x.expand(z.shape[0],*x.shape)   # Expand sequence data to have shape (batch_dim, sequence_length, *)
        t_input = torch.cat((x.detach(), z.detach()), dim=-1)
        t_input_perm = torch.cat((x.detach(), z_permuted.detach()), dim=-1)
        t_loss = -(self._t_network(t_input) - self._t_network(t_input_perm)).squeeze(dim=-1)
        self._t_optimizer.zero_grad()
        t_loss.mean().backward()
        self._t_optimizer.step()

        # > Mutual information < (need to compute this again because of backwards-tree)
        x = torch.cat((context['observations'], context['actions'], context['rewards'], context['next_observations'], context['terminals']), dim=-1)
        x = x.reshape((x.shape[0], -1)) # Stack sequence data to have shape (batch_dim, *)
        x = x.expand(z.shape[0],*x.shape)   # Expand sequence data to have shape (batch_dim, sequence_length, *)
        t_input = torch.cat((x, z), dim=-1)
        t_input_perm = torch.cat((x, z_permuted), dim=-1)
        t_diff = (self._t_network(t_input) - self._t_network(t_input_perm)).squeeze(dim=-1)

        # Accumulate ELBO
        elbo = reward_loglikelihood + obs_loglikelihood - self.beta * (z_logposterior - z_logprior)
        elbo = (elbo - self._alpha * t_diff).mean(dim=0)

        return elbo
