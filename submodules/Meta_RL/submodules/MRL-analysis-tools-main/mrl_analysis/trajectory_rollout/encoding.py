import os
from typing import List, Dict, Tuple, Any

import numpy as np
import random
from tqdm import tqdm

from ..utility.interfaces import MdpEncoder, MetaEnv, MetaRLPolicy
from .rollout import rollout_with_encoder
from .path_collector import MultithreadedPathCollector, MdpPathCollector
from smrl.utility.console_strings import print_to_terminal


def encodings_from_encoder(
    encoder: MdpEncoder,
    policy: MetaRLPolicy,
    env: MetaEnv,
    n_trajectories: int = 250,
    max_path_length: int = 250,
    encodings_per_trajectory: int = 10,
    randomize_samples: bool = False,
) -> Tuple[np.ndarray, List[Any], Dict, List[Dict[str, np.ndarray]]]:
    """Generate encodings from rollout trajectories.

    The rollouts use a given environment, policy, and encoder.

    Parameters
    ----------
    encoder : MdpEncoder
        Encoder, used for rollouts and encoding
    policy : MetaRLPolicy
        Policy, used for rollouts
    env : MetaEnv
        Environment, used for rollouts
    n_trajectories : int, optional
        Number of trajectories which are rolled out, by default 250
    max_path_length : int, optional
        Maximum length of a single trajectory, by default 250
    encodings_per_trajectory : int, optional
        Number of encodings generated from each trajectory, by default 10
    randomize_samples : bool, optional
        Set to True if you want to choose random samples from the same trajectory
        which are passed on to the encoder. In particular, these samples are not
        ordered by time any more and they can be sampled from any timesteps within
        a trajectory.

    Returns
    -------
    Tuple[np.ndarray, List[Any], Dict, List[Dict[str, np.ndarray]]]
        Encodings
        Tasks
        Contexts
        Trajectories
    """
    rollout_fn = rollout_with_encoder(encoder, context_size=encoder.context_size)

    path_collector_type = MdpPathCollector
    if 'MULTITHREADING' in os.environ.keys() and os.environ['MULTITHREADING'] == "True":
        path_collector_type = MultithreadedPathCollector
    path_collector = path_collector_type(env, policy, rollout_fn)
    
    trajectories = path_collector.collect_new_paths(max_path_length, n_trajectories)
    encodings, tasks, contexts = encodings_from_trajectories(
        encoder, trajectories, env, randomize=[], encodings_per_trajectory=encodings_per_trajectory, randomize_samples=randomize_samples
    )
    return encodings, tasks, contexts, trajectories


def encodings_from_trajectories(
    encoder: MdpEncoder,
    trajectories: List[Dict],
    env: MetaEnv,
    randomize: List[str] = None,
    encodings_per_trajectory: int = 10,
    randomize_samples: bool = False,
) -> Tuple[np.ndarray, List[Any], Dict]:
    """Encode contexts from multiple trajectories.

    See ``encodings_from_trajectory`` for the single-trajectory function which
    this function builds on.

    Parameters
    ----------
    encoder : MdpEncoder
        Encoder used for encoding
    trajectory : List[Dict]
        Trajectory, potentially generated from ``rollout.rollout_with_encoder()``
    env : MetaEnv
        Environment, used for random sampling from observation and action space
    randomize : List[str], optional
        List of randomizations, can be an arbitrary subset of:
        ``['observations', 'actions', 'rewards', 'next_observations', 'terminals']``,
        by default ``[]``
    encodings_per_trajectory : int, optional
        Number of encodings generated from each trajectory, by default 10
    randomize_samples : bool, optional
        Set to True if you want to choose random samples from the same trajectory
        which are passed on to the encoder. In particular, these samples are not
        ordered by time any more and they can be sampled from any timesteps within
        a trajectory.

    Returns
    -------
    Tuple[np.ndarray, List[Any], Dict]
        Encodings of shape (batch_size, context_size, encoding_dim)
        Tasks (ground truth)
        Context data (arrays of shape (batch_size, context_size, *))
    """
    encoding_data = []
    for traj in tqdm(trajectories):
        e = encodings_from_trajectory(
            encoder, traj, env=env, randomize=randomize, number_of_encodings=encodings_per_trajectory, randomize_samples=randomize_samples
        )
        if e is not None:
            encoding_data.append(e)
    encodings, tasks, contexts = zip(*encoding_data)
    encodings_stacked = np.array([encoding for encodings_traj in encodings for encoding in encodings_traj])
    tasks_stacked = [task for tasks_traj in tasks for task in tasks_traj]
    contexts_stacked = {}
    for key in contexts[0].keys():
        contexts_stacked[key] = np.concatenate([context[key] for context in contexts], axis=0)
    return encodings_stacked, tasks_stacked, contexts_stacked

def encodings_from_trajectory(
        encoder: MdpEncoder,
        trajectory: Dict,
        env: MetaEnv,
        randomize: List[str] = None,
        number_of_encodings: int = 10,
        randomize_samples: bool = False,
    ) -> Tuple[np.ndarray, List[Any], Dict]:
    """Encode contexts from a single trajectory.

    Parameters
    ----------
    encoder : MdpEncoder
        Encoder used for encoding
    trajectory : Dict
        Trajectory, potentially generated from ``rollout.rollout_with_encoder()``
    env : MetaEnv
        Environment, used for random sampling from observation and action space
    randomize : List[str], optional
        List of randomizations, can be an arbitrary subset of:
        ``['observations', 'actions', 'rewards', 'next_observations', 'terminals']``,
        by default ``[]``
    number_of_encodings : int, optional
        Number of encodings generated from the trajectory, by default 10
    randomize_samples : bool, optional
        Set to True if you want to choose random samples from the same trajectory
        which are passed on to the encoder. In particular, these samples are not
        ordered by time any more and they can be sampled from any timesteps within
        a trajectory.

    Returns
    -------
    Tuple[np.ndarray, List[Any], Dict]
        Encodings of shape (batch_size, context_size, encoding_dim)
        Tasks (ground truth)
        Context data (arrays of shape (batch_size, context_size, *))
    """

    # Input parsing
    if randomize is None:
        randomize = []
    admissible_values = ['observations', 'actions', 'rewards', 'next_observations', 'terminals']
    for item in randomize:
        if item not in admissible_values:
            raise ValueError(f"Unknown value '{item}' in argument ``randomize``!")

    # Randomize inputs according to inputs
    traj_length = len(trajectory['observations'])
    if 'observations' in randomize:
        trajectory['observations'] = np.array([env.observation_space.sample() for _ in range(traj_length)])
    if 'actions' in randomize:
        trajectory['actions'] = np.array([env.action_space.sample() for _ in range(traj_length)])
    if 'rewards' in randomize:
        trajectory['rewards'] = np.array([[-random.random()] for _ in range(traj_length)])
    if 'next_observations' in randomize:
        trajectory['next_observations'] = np.array([env.observation_space.sample() for _ in range(traj_length)])
    if 'terminals' in randomize:
        trajectory['terminals'] = np.random.rand(*trajectory['terminals'].shape) > 0.5

    # Get trajectory data of this trajectory
    if trajectory['observations'].shape[0] < encoder.context_size:
        print(f"Trajectory is not long enough for history length.")
        return None
    obs_batch, act_batch, rew_batch, next_obs_batch, term_batch, tasks = \
        encoder_batch_from_trajectory(
            trajectory, encoder.context_size, number_of_batches=number_of_encodings, randomize_samples=randomize_samples
        )

    # Encode contexts
    encodings = encoder.get_encodings(obs_batch, act_batch, rew_batch, next_obs_batch, term_batch)
    context = dict(
        observations=obs_batch,
        actions=act_batch,
        rewards=rew_batch,
        next_observations=next_obs_batch,
        terminals=term_batch,
    )
    
    return encodings, tasks, context


def encoder_batch_from_trajectory(trajectory: Dict, context_size: int, number_of_batches: int, randomize_samples: bool) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create encoder input batch from trajectory data.

    Each entry in the batch has shape (n, context_size, *),
        where n = len(list(range(context_size, observations.shape[0], step)))
    i.e. we take every step'th context, beginning at t = context_size.

    Parameters
    ----------
    trajectory : Dict
        Dictionary of trajectory data with numpy array entries 'observations', 
        'actions', 'rewards', 'next_observations', and 'terminals'
    context_size : int
        Length of context sequences
    number_of_batches : int
        Number of batches to be sampled
    randomize_samples : bool
        Set to False if you want to get samples which are subsequent and in time
        ordering.
        Set to True if you want to choose random samples from the same trajectory
        which are passed on to the encoder. In particular, these samples are not
        ordered by time any more and they can be sampled from any timesteps within
        a trajectory.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Batches of shape (n, context_size, *)
        - observations
        - actions
        - rewards
        - next observations
        - terminals
    """

    observations = trajectory['observations']
    actions = trajectory['actions']
    rewards = trajectory['rewards']
    next_observations = trajectory['next_observations']
    terminals = trajectory['terminals']
    tasks = np.array(trajectory['tasks'])

    if randomize_samples:
        idxs = np.random.choice(len(observations), size=(number_of_batches, context_size), replace=True)
    else:
        times = np.linspace(0, len(observations) - context_size, number_of_batches).astype(int)
        idxs = times[..., None] + np.arange(context_size)

    obs_batch = observations[idxs]
    act_batch = actions[idxs]
    rew_batch = rewards[idxs]
    next_obs_batch = next_observations[idxs]
    term_batch = terminals[idxs]
    tasks = tasks[idxs[:,-1]]

    return obs_batch, act_batch, rew_batch, next_obs_batch, term_batch, tasks

def contexts_from_trajectory(trajectory: Dict, context_size: int) -> Dict:
    """
    Create contexts from trajectory data.

    Each context combines the last ``context_size`` timesteps. If some timesteps
    are unavailable (at trajectory start), the sequence is padded with zeros.

    Parameters
    ----------
    trajectory : Dict
        Trajectory dictionary with keys
        'observations', np.ndarray with shape (trajectory_length, *)
        'actions', np.ndarray with shape (trajectory_length, *)
        'rewards', np.ndarray with shape (trajectory_length, *)
        'next_observations', np.ndarray with shape (trajectory_length, *)
        'terminals', np.ndarray with shape (trajectory_length, *)
    context_size : int
        Length of each context sequence

    Returns
    -------
    Dict
        Context dictionary with keys
            'observations', np.ndarray with shape (trajectory_length, context_size, *)
            'actions', np.ndarray with shape (trajectory_length, context_size, *)
            'rewards', np.ndarray with shape (trajectory_length, context_size, *)
            'next_observations', np.ndarray with shape (trajectory_length, context_size, *)
            'terminals', np.ndarray with shape (trajectory_length, context_size, *)
    """
    trajectory_length = trajectory['observations'].shape[0]
    times = np.arange(trajectory_length)
    
    indices = times[..., None] - np.arange(context_size, 0, -1)[None, ...]

    contexts = {
        'observations': np.zeros((trajectory_length, context_size, *trajectory['observations'].shape[1:])),
        'actions': np.zeros((trajectory_length, context_size, *trajectory['actions'].shape[1:])),
        'rewards': np.zeros((trajectory_length, context_size, *trajectory['rewards'].shape[1:])),
        'next_observations': np.zeros((trajectory_length, context_size, *trajectory['next_observations'].shape[1:])),
        'terminals': np.zeros((trajectory_length, context_size, *trajectory['terminals'].shape[1:])),
    }

    for i, context_times in enumerate(indices):
        context_times = context_times[context_times >= 0]
        for key in ['observations', 'actions', 'rewards', 'next_observations', 'terminals']:
            data: np.ndarray = trajectory[key][context_times]
            contexts[key][i] = np.pad(data, ((context_size - len(data), 0), *[(0, 0) for _ in range(data.ndim - 1)]))
    # contexts['tasks'] = trajectory['tasks']
    return contexts