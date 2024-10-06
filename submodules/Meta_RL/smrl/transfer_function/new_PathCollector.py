from mrl_analysis.utility import interfaces, pytorch_util as ptu
import os
import numpy as np
import torch
from typing import Dict, List, Union, Any, Tuple
from tqdm import tqdm
import copy

from main_config import HARDCODED
from smrl.policies.meta_policy import PretrainedCheetah

from mrl_analysis.trajectory_rollout.path_collector import MdpPathCollector, MultithreadedPathCollector
from new_rollout import rollout_with_encoder
from mrl_analysis.trajectory_rollout.encoding import contexts_from_trajectory

class TrajectoryGeneratorWithTransferFunction():
    """
    This class can generate trajectories with additional information for analysis.

    Use the method ``run()`` to generate trajectories.
    """

    def __init__(
            self,
            simple_env: interfaces.MetaEnv,
            env: interfaces.MetaEnv, 
            policy: interfaces.MetaRLPolicy, 
            encoder: interfaces.MdpEncoder, 
            transfer_function,
            decoder: interfaces.MdpDecoder = None,
            qfunction: interfaces.MetaQFunction = None,
        ):
        self.simple_env = simple_env
        self.env = env
        self.policy = policy
        self.encoder = encoder
        self.decoder = decoder
        self.qfunction = qfunction
        self.transfer_function= transfer_function

    def run(self, n_trajectories: int, max_path_length: int) -> List[Dict[str, Union[np.ndarray, Any]]]:
        """
        Generates trajectory data by rollouts with additional information.

        Set ``os.environ['MULTITHREADING'] = "True"`` to accelerate rollouts with
        ray.

        Parameters
        ----------
        n_trajectories : int
            Number of trajectories
        max_path_length : int
            Maximum length of a single trajectory
        
        Returns
        -------
        List[Dict[str, Union[np.ndarray, Any]]]
            A list of path dictionaries.
            Each dictionary stands for one trajectory and contains the following keys:
            - 'observations'
            - 'actions'
            - 'rewards'
            - 'next_observations'
            - 'dones'
            - 'contexts' : Dict[str, np.ndarray], The contexts
            - 'action_dist': Distribution, Distribution over actions
            - 'action_mean': Mean of action distribution
            - 'action_std': Standard deviation of action distribution
            - 'latent_dist': Distribution, Distribution over the latent variables
            - 'latent_mean': Mean of latent variables' distribution
            - 'latent_std': Standard deviation of latent distribution
            - 'qvalues': Q-function prediction (only if qfunction provided)
            - 'rew_dist': Distribution over encoder-predicted rewards (only if decoder provided)
            - 'std_dist': Distribution over encoder-predicted next states (only if decoder provided)
        """

        rollout_fn = rollout_with_encoder(self.simple_env, self.encoder, self.transfer_function, context_size=self.encoder.context_size)
        path_collector_type = MdpPathCollector
        if 'MULTITHREADING' in os.environ.keys() and os.environ['MULTITHREADING'] == "True":
            path_collector_type = MultithreadedPathCollector
        path_collector = path_collector_type(self.env, self.policy, rollout_fn)
        
        print("Collecting trajectories...")
        trajectories = path_collector.collect_new_paths(max_path_length, n_trajectories)

        print("Adding additional information...")
        for traj in tqdm(trajectories):
            contexts = self.contexts(traj)
            traj['contexts'] = contexts
            action_dist = self.action_distribution(traj)
            traj['action_dist'] = action_dist
            traj['action_mean'] = ptu.np_ify(action_dist.mean)
            traj['action_std'] = np.sqrt(ptu.np_ify(action_dist.variance))
            latent_dist = self.latent_distribution(contexts)
            traj['latent_dist'] = latent_dist
            traj['latent_mean'] = ptu.np_ify(latent_dist.mean)
            traj['latent_std'] = ptu.np_ify(latent_dist.stddev)

            if self.decoder is not None:
                rew_dist, state_dist = self.decoder_distributions(traj, traj['latent_mean'])
                traj['rew_dist'] = rew_dist
                traj['state_dist'] = state_dist
                traj['rew_pred_mean'] = ptu.np_ify(rew_dist.mean)
                traj['rew_pred_std'] = ptu.np_ify(rew_dist.stddev)
                traj['state_pred_mean'] = ptu.np_ify(state_dist.mean)
                traj['state_pred_std'] = ptu.np_ify(state_dist.stddev)

            if self.qfunction is not None:
                traj['qvalues'] = self.qvalues(traj)

        return trajectories

    def contexts(self, trajectory: Dict) -> Dict[str, np.ndarray]:
        return contexts_from_trajectory(trajectory, self.encoder.context_size)

    def action_distribution(self, trajectory: Dict[str, np.ndarray]) -> torch.distributions.Distribution:
        return self.policy.forward(
            ptu.torch_ify(trajectory['observations'][:, 0]).unsqueeze(dim=1), 
            ptu.torch_ify(trajectory['encodings'])
        )

    def latent_distribution(self, contexts: Dict[str, np.ndarray]) -> torch.distributions.Distribution:
        return self.encoder.forward(**ptu.np_batch_to_tensor_batch(copy.deepcopy(contexts)))

    def decoder_distributions(self, trajectory: Dict, latent_mean: np.ndarray) -> Tuple[torch.distributions.Distribution, torch.distributions.Distribution]:
        return self.decoder.forward(
            ptu.torch_ify(latent_mean),
            ptu.torch_ify(trajectory['observations']),
            ptu.torch_ify(trajectory['actions'])
        )

    def qvalues(self, trajectory: Dict[str, np.ndarray]) -> np.ndarray:
        return ptu.np_ify(
            self.qfunction.forward(
                ptu.torch_ify(trajectory['observations']),
                ptu.torch_ify(trajectory['actions']),
                ptu.torch_ify(trajectory['encodings'])
            )
        )