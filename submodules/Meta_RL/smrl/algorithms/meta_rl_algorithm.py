"""
This module contains the training algorithm class `MetaRlAlgorithm` for Meta-RL agents.
The algorithm simultaneously trains an encoder-decoder network (usually a VAE)
and a policy.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-27
"""

import torch
import gtimer
from tqdm import tqdm

from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchTrainer
from rlkit.samplers.data_collector import PathCollector
from rlkit.data_management.replay_buffer import ReplayBuffer

import smrl.utility.console_strings as console_strings
from smrl.utility.console_strings import print_to_terminal
from smrl.utility.device_context_manager import DeviceContext
from smrl.environments.meta_env import MetaEnv
from smrl.data_management.replay_buffers import ContextReplayBuffer

from .multi_wrappers import *


class MetaRlAlgorithm(TorchBatchRLAlgorithm):
    """An algorithm for simultaneous training of
    - an inference mechanism (e.g. VAE)
    - a policy

    Each of the above components has its own trainer (`TorchTrainer`).

    Be aware that this algorithm uses batches for training, i.e. it only works
    for offline RL methods.
    
    In each epoch, the algorithm 
    1. samples evaluation trajectories (once),
    2. samples exploration trajectories and adds them to the buffer (multiple times),
    3. trains the policy (multiple times),
    4. trains the inference mechanism (multiple times).

    One epoch consists of `num_train_loops_per_epoch` training loops.
    Each training loop can have multiple training steps for the inference mechanism
    and policy.

    For more details see `_train()`.

    Parameters
    ----------
    policy_trainer : TorchTrainer
        The trainer for the policy. Should respect the latent input of the policy.
        See `MetaSACTrainer` in smrl/trainers/meta_sac.py for an example.
    inference_trainer : TorchTrainer
        The trainer for the inference mechanism (VAE).
        See `MdpVAETrainer` in smrl/trainers/vae.py for an example.
    exploration_env : MetaEnv
        Exploration environment
    evaluation_env : MetaEnv
        Evaluation environment
    exploration_data_collector : PathCollector
        Data collector for policy training samples.
    inference_data_collector : PathCollector
        Data collector for inference training samples.
    evaluation_data_collector : PathCollector
        Data collector for evaluation purposes.
    expl_replay_buffer : ReplayBuffer
        Replay buffer for policy batch learning
    inference_replay_buffer : ContextReplayBuffer
        Replay buffer for inference batch learning
    batch_size : int
        Size of randomized batches
    context_size : int
        Lenght of context sequence for each sample
    prediction_target_size : int
        Size of the prediction targets for ELBO computation
    max_path_length : int
        Maximum length of a single trajectory
    num_epochs : int
        Number of training epochs
    num_eval_paths_per_epoch : int
        Number of evaluation paths (trajectories!) in each epoch.
        Each epoch evaluates the performance based on newly created samples.
    num_expl_paths_per_epoch : int
        Number of exploration paths (trajectories!) which is sampled
        and added to the policy replay buffer in each epoch.
    num_inference_paths_per_epoch : int
        Number of exploration paths (traectories!) which is sampled and added
        to the inference replay buffer in each epoch
    num_inference_trains_per_train_loop : int
        Number of training steps for the inference mechanism in each training loop.
    num_policy_trains_per_train_loop : int
        Number of training steps for the policy in each training loop.
    num_train_loops_per_epoch : int, optional
        Number of training loops per epoch, by default 1
    min_num_steps_before_training : int, optional
        Number of samples (transitions) collected and added to the replay buffer
        before actual training starts, by default 0
    start_epoch : int, optional
        Specify that this epoch is not the first (e.g. if the model is loaded from
        an already trained model and training is continued), by default 0
    """
    def __init__(
            self,
            policy_trainer: TorchTrainer,
            inference_trainer: TorchTrainer,
            exploration_env: MetaEnv,
            evaluation_env: MetaEnv,
            exploration_data_collector: PathCollector,
            inference_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            expl_replay_buffer: ReplayBuffer,
            inference_replay_buffer: ContextReplayBuffer,
            batch_size: int,
            context_size: int,
            prediction_target_size: int,
            max_path_length: int,
            num_epochs: int,
            num_eval_paths_per_epoch: int,
            num_expl_paths_per_epoch: int,
            num_inference_paths_per_epoch: int,
            num_inference_trains_per_train_loop: int,
            num_policy_trains_per_train_loop: int,
            num_train_loops_per_epoch: int = 1,
            min_num_steps_before_training: int = 0,
            start_epoch: int = 0,
        ):

        self.context_size = context_size
        self.prediction_target_size = prediction_target_size

        self.num_inference_trains_per_train_loop = num_inference_trains_per_train_loop
        self.num_policy_trains_per_train_loop = num_policy_trains_per_train_loop
        self.num_expl_paths_per_epoch = num_expl_paths_per_epoch
        self.num_inference_paths_per_epoch = num_inference_paths_per_epoch
        self.num_eval_paths_per_epoch = num_eval_paths_per_epoch

        self.policy_trainer = policy_trainer
        self.inference_trainer = inference_trainer

        self.policy_data_collector = exploration_data_collector
        self.inference_data_collector = inference_data_collector

        self.expl_replay_buffer = expl_replay_buffer
        self.inference_replay_buffer = inference_replay_buffer

        # Multi-object wrappers
        trainer = MultiTrainerWrapper({
            "Inference trainer": self.inference_trainer,
            "Policy trainer": self.policy_trainer
        })
        data_collector = MultiCollectorWrapper({
            'Inference': self.inference_data_collector,
            'Exploration': self.policy_data_collector,
        })
        buffer = MultiBufferWrapper({
            'Policy': expl_replay_buffer,
            'Inference': inference_replay_buffer,
        })

        # super().__init__()
        super().__init__(
            trainer=trainer,
            exploration_env=exploration_env,
            evaluation_env=evaluation_env,
            exploration_data_collector=data_collector,
            evaluation_data_collector=evaluation_data_collector,
            replay_buffer=buffer,
            batch_size=batch_size,
            max_path_length=max_path_length,
            num_epochs=num_epochs,
            num_eval_steps_per_epoch=None,
            num_expl_steps_per_train_loop=None,
            num_trains_per_train_loop=0, # We have two separate variables instead
            num_train_loops_per_epoch=num_train_loops_per_epoch,
            min_num_steps_before_training=min_num_steps_before_training,
            start_epoch=start_epoch,
        )

        gtimer.reset_root() # Reset gtimer to allow multiple subsequent calls of the algorithm

    def _train(self):
        """Train the inference mechanism (VAE) and the policy for one epoch.
        
        One epoch can comprise multiple training steps. An epoch performs the following steps:
        1. Sample paths for evaluation
        2. Repeat for some steps (`num_train_loops_per_epoch`):
            1. Sample new paths for exploration and add them to the buffer
            2. Train policy for some steps (`num_policy_trains_per_train_loop`)
            3. Train inference mechanism for some steps (`num_inference_trains_per_train_loop`)
        """

        print_to_terminal(console_strings.bold(f"\nEpoch {self.epoch}"))

        self.training_mode(False)

        # Perform rollouts on cpu
        with DeviceContext(torch.device('cpu'), self, verbose=console_strings.verbose):

            # Initial exploration trajectories (optional)
            if self.epoch == 0 and self.min_num_steps_before_training > 0:
                print_to_terminal("Collecting initial samples ...")
                init_expl_paths = self.policy_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.min_num_steps_before_training,
                    discard_incomplete_paths=False,
                )
                if not self.offline_rl:
                    self.expl_replay_buffer.add_paths(init_expl_paths)
                self.policy_data_collector.end_epoch(-1)
            gtimer.stamp('initial sampling', unique=False)

            # Evaluation trajectories
            print_to_terminal("Collecting evaluation samples ...")
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_paths_per_epoch,
                discard_incomplete_paths=True,
            )
            gtimer.stamp('evaluation sampling', unique=False)

            # New exploration paths
            print_to_terminal("Collecting exploration samples ...")
            new_expl_paths = self.policy_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_paths_per_epoch,
                discard_incomplete_paths=False,
            )
            print_to_terminal("Collecting inference exploration samples ...")
            new_inference_paths = self.inference_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_inference_paths_per_epoch,
                discard_incomplete_paths=False,
            )
            gtimer.stamp('exploration sampling', unique=False)

            if not self.offline_rl:
                self.expl_replay_buffer.add_paths(new_expl_paths)
                self.inference_replay_buffer.add_paths(new_inference_paths)
            gtimer.stamp('data storing', unique=False)


        # Train loop with batch training
        print_to_terminal("Training ...")
        self.training_mode(True)

        iterator = tqdm(range(self.num_train_loops_per_epoch), disable=(not console_strings.verbose))
        for _ in iterator:

            # Train policy
            gtimer.subdivide('policy training')
            for _ in range(self.num_policy_trains_per_train_loop):
                iterator.set_description(f"Training (policy)")
                train_data = self.expl_replay_buffer.random_batch(
                    self.batch_size, self.context_size
                )
                self.policy_trainer.train(train_data)
            gtimer.end_subdivision()
            gtimer.stamp('policy training', unique=False)

            # Train inference mechanism
            gtimer.subdivide('inference training')
            for _ in range(self.num_inference_trains_per_train_loop):
                iterator.set_description(f"Training (inference)")
                train_data = self.inference_replay_buffer.random_context_target_batch(
                    self.batch_size, self.context_size, self.prediction_target_size
                )
                self.inference_trainer.train(train_data)
            gtimer.end_subdivision()
            gtimer.stamp('inference training', unique=False)

        self.training_mode(False)

    def _get_snapshot(self):
        snapshot = super()._get_snapshot()
        snapshot.update({'epoch': self.epoch})
        return snapshot
