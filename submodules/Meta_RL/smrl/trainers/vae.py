"""
This module contains `MdpVAETrainer`, a class which is used to train `MdpVAE` instances.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2022-12-15
"""

import torch
import torch.nn.functional as F
import gtimer
from typing import Iterable, Dict, Tuple, Callable
from collections import OrderedDict

from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.loss import LossFunction

from ..vae.mdpvae import MdpVAE
from ..utility.ops import np_batch_to_tensor_batch

class MdpVAETrainer(TorchTrainer, LossFunction):
    """A trainer for MdpVAEs

    Parameters
    ----------
    vae : MdpVAE
        The VAE to train
    lr : float, optional
        Learning rate of the optimizer, by default 1e-3
    n_latent_samples : int, optional
        Number of latent samples for Monte-Carlo estimation of the ELBO, by default 100
    beta_schedule : Callable[[int], float], optional
        Function which determines the ELBO's beta-value for each step.
        See Fu et al. (2019) "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing".
        By default None
    optimizer_class : type[torch.optim.Optimizer], optional
        The optimizer (class) for the VAE parameters, by default torch.optim.Adam
    clipping : float, optional
        Clipping value for losses (only before backward()-call). If None is passed,
        clipping will be set to torch.inf.
        By default 1e3
    """
    def __init__(
            self,
            vae: MdpVAE,
            lr: float = 1e-3,
            n_latent_samples: int = 100,
            beta_schedule: Callable[[int], float] = None,
            optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
            clipping: float = None,
        ):
        super().__init__()
        self.vae = vae
        self.n_latent_samples = n_latent_samples
        self.beta_schedule = beta_schedule

        self.optimizer = optimizer_class(
            self.vae.parameters(),
            lr = lr
        )
        self.clipping = clipping if clipping is not None else torch.inf

        self.eval_statistics = OrderedDict()

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_batch_to_tensor_batch(np_batch)
        self.train_from_torch(batch)

    def train_from_torch(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        """Train VAE for one step.

        Parameters
        ----------
        batch : Dict[str, Dict[str, torch.Tensor]]
            Training batch with entries:
                'context'
                'target'
                    each of them having entries:
                    'observations'
                    'actions'
                    'rewards'
                    'next_observations'
                    'terminals'
        """
        gtimer.blank_stamp()
        
        loss, stats = self.compute_loss(
            batch,
            skip_statistics=False,
        )
        self.eval_statistics = stats

        self.optimizer.zero_grad()
        loss.clip(-self.clipping, self.clipping).backward()
        self.optimizer.step()

        if self.beta_schedule is not None:
            self.vae.beta = self.beta_schedule(self._num_train_steps)

        gtimer.stamp('vae training', unique=False)


    def compute_loss(
            self,
            batch: Dict[str, Dict[str, torch.Tensor]],
            skip_statistics: bool = False
        ) -> Tuple[torch.Tensor, OrderedDict]:
        """Compute the training loss.

        Parameters
        ----------
        batch : Dict[str, Dict[str, torch.Tensor]]
            Training batch with entries:
                'context'
                'target'
                    each of them having entries:
                    'observations'
                    'actions'
                    'rewards'
                    'next_observations'
                    'terminals'
        skip_statistics : bool, optional
            If True, the returned statistics are empty, by default False

        Returns
        -------
        Tuple[torch.Tensor, OrderedDict]
            Loss, Statistics
        """
        elbo = self.vae.elbo(
            **batch,
            n_latent_samples=self.n_latent_samples,
        ).mean()

        stats = OrderedDict()
        if not skip_statistics:
            latents = self.vae.encoder.forward(**batch['context']).mean
            latents = latents.unsqueeze(1).repeat(1, batch['target']['observations'].shape[1], 1)
            predicted_rewards, predicted_next_states = self.vae.decoder.forward(latents, batch['target']['observations'], batch['target']['actions'])
            stats.update({
                'elbo': elbo.item(),
                'reward prediction error': F.l1_loss(predicted_rewards.mean, batch['target']['rewards']).detach().item(),
                'state prediction error': F.l1_loss(predicted_next_states.mean, batch['target']['next_observations']).detach().item(),
                'beta': self.vae.beta,
            })

        # Loss for training is the *negative* ELBO
        return -elbo, stats

    @property
    def networks(self) -> Iterable[torch.nn.Module]:
        return [self.vae]

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        # stats.update({  # DEBUGGING, activate if necessary (e.g. to evaluate trained decoder standard deviations)
        #     'std_rew': self.vae.decoder._std_rew.cpu().detach().numpy(),
        #     'std_obs': self.vae.decoder._std_obs.cpu().detach().numpy()
        # })
        return stats
    
    def get_snapshot(self):
        return dict(
            # vae=self.vae,
            encoder=self.vae.encoder.state_dict(),
            decoder=self.vae.decoder.state_dict(),
        )