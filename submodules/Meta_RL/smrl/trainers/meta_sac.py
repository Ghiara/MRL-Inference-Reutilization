"""
This module contains the class MetaSACTrainer which is derived from
rlkit's SACTrainer. It works like the original SAC-trainer but extends
its functionality to meta-RL policies and meta-RL value functions by
using and encoder.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-22
"""

import torch
import gym
import numpy as np
import gtimer

from collections import OrderedDict
from typing import Tuple, Dict, Any

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.sac import SACTrainer, SACLosses
from rlkit.core.loss import LossStatistics
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.logging import add_prefix

from smrl.policies.base import MetaRLPolicy, MetaQFunction
from smrl.vae.mdpvae import MdpEncoder
from smrl.utility.ops import np_batch_to_tensor_batch

class MetaSACTrainer(SACTrainer):
    """A trainer for SAC networks and an MDP-VAE policy.

    This class is derived from rlkit's `SACTrainer` and extends its
    functionality by an encoder which computes additional input for
    the policy based on context information.

    The trainer uses two Q-function networks and target-networks to
    stabilize training.

    Parameters
    ----------
    env : gym.Env
        Environment, used for determining the dimensionality of
        the observation and action spaces.
    policy : MetaRLPolicy
        A meta-rl policy which takes an encoded context variable
        as additional input, aside from the usual state input.
        It returns a distribution over the action space.
    encoder : MdpEncoder
        An encoder used for mapping context data (e.g. transition
        history) to a encodings which describe the task.
    qf1 : MetaQFunction
        First Q-function network.
    qf2 : MetaQFunction
        Second Q-function network.
    target_qf1 : MetaQFunction
        First target Q-function network.
    target_qf2 : MetaQFunction
        Second target Q-function network.
    discount : float, optional
        Discount factor of the MDP, by default 0.99
    reward_scale : float, optional
        Scales the reward function for learning the Q-function, by default 1
    policy_lr : float, optional
        Learning rate for the policy, by default 0.001
    qf_lr : float, optional
        Learning rate for the Q-function, by default 0.001
    encoder_lr : float, optional
        Learning rate for the encoder (Encoder will be trained with policy loss),
        If ``None``, the encoder will not be trained,
        by default None
        NOTE: This functionality has not been tested!
    optimizer_class : type[torch.optim.Optimizer], optional
        Optimizer used to train policy and Q-function networks,
        by default torch.optim.Adam
    soft_target_tau : float, optional
        Parameter for the soft updates of the target Q-function networks
        The two extreme values are:
            1.0: target network is fully replaced by the Q-network
            0.0: target network is not updated at all, 
        by default 0.01
    target_update_period : int, optional
        Number of training steps after which the target networks are updated,
        by default 1
    use_automatic_entropy_tuning : bool, optional
        If set to True, the entropy temperature alpha is also optimized,
        by default True
    target_entropy : float, optional
        Target entropy used for computing the entropy-loss. 
        This parameter is only relevant when `use_automatic_entropy_tuning=True`.
        By default, a heuristic based on the dimensionality of the 
        action space is used.
        By default None
    clipping : float, optional
        Clipping value for losses (only before backward()-call). If None is passed,
        clipping will be set to torch.inf.
        By default 1e3
    """
    def __init__(
        self,
        env: gym.Env,
        policy: MetaRLPolicy,
        encoder: MdpEncoder,
        qf1: MetaQFunction,
        qf2: MetaQFunction,
        target_qf1: MetaQFunction,
        target_qf2: MetaQFunction,
        discount: float = 0.99,
        reward_scale: float = 1.0,
        policy_lr: float = 0.001,
        qf_lr: float = 0.001,
        encoder_lr: float = None,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        soft_target_tau: float = 0.01,
        target_update_period: int = 1,
        use_automatic_entropy_tuning: bool = True,
        target_entropy: float = None,
        clipping: float = 1e3,
        **kwargs,
    ):
        super().__init__(env, policy, qf1, qf2, target_qf1, target_qf2, 
                        discount=discount, reward_scale=reward_scale, 
                        policy_lr=policy_lr, qf_lr=qf_lr, 
                        optimizer_class=optimizer_class,
                        soft_target_tau=soft_target_tau, 
                        target_update_period=target_update_period, 
                        use_automatic_entropy_tuning=use_automatic_entropy_tuning, 
                        target_entropy=target_entropy, **kwargs)
        self.encoder = encoder
        self.policy: MetaRLPolicy
        self.qf1: MetaQFunction
        self.qf2: MetaQFunction
        self.target_qf1: MetaQFunction
        self.target_qf2: MetaQFunction
        self.clipping = clipping if clipping is not None else torch.inf

        self._train_encoder = False
        if encoder_lr is not None:
            self._train_encoder = True
            self.encoder_optimizer = optimizer_class(
                self.encoder.parameters(),
                lr = encoder_lr,
            )

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_batch_to_tensor_batch(np_batch)
        self.train_from_torch(batch)

    def train_from_torch(self, batch):
        gtimer.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )

        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.clip(-self.clipping, self.clipping).backward()
            self.alpha_optimizer.step()

        if self._train_encoder:
            self.encoder_optimizer.zero_grad()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.clip(-self.clipping, self.clipping).backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.clip(-self.clipping, self.clipping).backward(retain_graph=self._train_encoder)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.clip(-self.clipping, self.clipping).backward(retain_graph=self._train_encoder)
        self.qf2_optimizer.step()

        if self._train_encoder:
            self.encoder_optimizer.step()
            

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gtimer.stamp('sac training', unique=False)

    def compute_loss(
            self,
            batch: Dict,
            skip_statistics: bool = False,
        ) -> Tuple[SACLosses, LossStatistics]:
        """Compute the following losses for a batch of training data:
        - Policy loss
        - Q-function losses
        - Entropy loss

        The batch is a dictionary with following entries:
        - "observations" : torch.Tensor, shape (batch_size, observation_dim)
        - "actions" : torch.Tensor, shape (batch_size, action_dim)
        - "rewards" : torch.Tensor, shape (batch_size, 1)
        - "next_observations" : torch.Tensor, shape (batch_size, observation_dim)
        - "context" : Dict
            Dictionary with entries
            - "observations" : torch.Tensor, shape (batch_size, context_size, observation_dim)
            - "actions" : torch.Tensor, shape (batch_size, context_size, action_dim)
            - "rewards" : torch.Tensor, shape (batch_size, context_size, 1)
            - "next_observations" : torch.Tensor, shape (batch_size, context_size, observation_dim)
            - "terminals" : torch.Tensor, shape (batch_size, context_size, 1)
        
        An example for creating such batches is given by `ContextReplayBuffer`
        and its function `random_batch` (directory 
        "smrl/data_management/replay_buffers/context_replay_buffer").
        
        The code is copied from rlkit/torch/sac/sac.py and modified to be
        compatible with MdpVAEs.

        Parameters
        ----------
        batch : Dict
            Dictionary with transition data. See above for required keywords and
            arguments.
        skip_statistics : bool, optional
            Set to True if the output should not contain training statistics,
            by default False

        Returns
        -------
        Tuple[SACLosses, LossStatistics]
            - Losses (Policy loss, Q-function losses, Entropy loss)
            - Training statistics
        """
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        context = batch['context']

        # Encode context data to obtain latent variables for this timestep AND
        # the *next* timestep
        next_context = dict(
            observations=torch.concat((context['observations'], obs[:,None,:]), dim=1),
            actions=torch.concat((context['actions'], actions[:,None,:]), dim=1),
            rewards=torch.concat((context['rewards'], rewards[:,None,:]), dim=1),
            next_observations=torch.concat((context['next_observations'], next_obs[:,None,:]), dim=1),
            terminals=torch.concat((context['terminals'], terminals[:,None,:]), dim=1),
        )

        if self._train_encoder:
            encoding = self.encoder.get_encodings(**context)
            next_encoding = self.encoder.get_encodings(**next_context)
        else:
            with torch.no_grad():
                encoding = self.encoder.get_encodings(**context).detach()
                next_encoding = self.encoder.get_encodings(**next_context).detach()
                # OPTION: Use same encodings that were used during sampling
                # encoding = batch['encodings']
                # next_encoding = batch['encodings']

        """
        Policy and Alpha Loss
        """
        dist = self.policy.forward(obs, encoding.detach())
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            if self.log_alpha.device != log_pi.device:
                self.log_alpha = self.log_alpha.to(log_pi.device)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1
        alpha = 0
        q_new_actions = torch.min(
            self.qf1.forward(obs, new_obs_actions, encoding.detach()),
            self.qf2.forward(obs, new_obs_actions, encoding.detach()),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions, encoding)
        q2_pred = self.qf2(obs, actions, encoding)
        next_dist = self.policy.forward(next_obs, next_encoding.detach())
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = torch.min(
            self.target_qf1.forward(next_obs, new_next_actions, next_encoding.detach()),
            self.target_qf2.forward(next_obs, new_next_actions, next_encoding.detach()),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Encoding',
                ptu.get_numpy(encoding),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Next encoding',
                ptu.get_numpy(next_encoding),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )

        return loss, eval_statistics

    def get_snapshot(self):
        return dict(
            policy=self.policy.state_dict(),
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            target_qf1=self.target_qf1.state_dict(),
            target_qf2=self.target_qf2.state_dict(),
        )
