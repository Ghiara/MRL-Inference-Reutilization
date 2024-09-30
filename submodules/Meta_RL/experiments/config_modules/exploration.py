from typing import Dict, Any, Type, Tuple

from smrl.policies.base import Policy
from smrl.policies.exploration import RandomPolicy, RandomMemoryPolicy, MultiRandomMemoryPolicy, LogMultiRandomMemoryPolicy


def _expl_policy(
    policy_type: Type[Policy],
    policy_kwargs: Dict[str, Any],
    inference_only: bool,
) -> Dict[str, Any]:
    """
    Helper function which maps policy configurations to the respective keys
    for random exploration or random inference exploration
    """
    if inference_only:
        return dict(
            inference_policy_type = policy_type,
            inference_policy_kwargs = policy_kwargs
        )
    else:
        return dict(
            expl_policy_type = policy_type,
            expl_policy_kwargs = policy_kwargs
        )

def random_policy(
    std: float,
    mean: float = 0.0,
    inference_only: bool = False,
) -> Dict[str, Any]:
    """
    Config for random exploration policy ``RandomPolicy``
    """
    kwargs = {
        'mean': mean,
        'std': std,
    }
    return _expl_policy(RandomPolicy, kwargs, inference_only=inference_only)
    

def random_memory_policy(
    action_update_interval: int,
    mean_std: float,
    std_mean: float,
    mean: float = 0.0,
    sample_update_interval: bool = False,
    inference_only: bool = False,
) -> Dict[str, Any]:
    """
    Config for random exploration policy ``RandomMemoryPolicy``
    """
    return _expl_policy(
        RandomMemoryPolicy,
        {
            'action_update_interval': action_update_interval,
            'mean_std': mean_std,
            'std_mean': std_mean,
            'mean': mean,
            'sample_update_interval': sample_update_interval,
        },
        inference_only=inference_only,
    )

def multi_random_memory_policy(
    action_update_interval: int,
    mean_std_range: Tuple[float, float],
    std_mean_range: Tuple[float, float],
    S: float,
    M: float = 0.0,
    sample_update_interval: bool = False,
    inference_only: bool = False,
) -> Dict[str, Any]:
    """
    Config for random exploration policy ``MultiRandomMemoryPolicy``
    """
    return _expl_policy(
        MultiRandomMemoryPolicy,
        {
            'action_update_interval': action_update_interval,
            'M': M,
            'S': S,
            'mean_std_range': mean_std_range,
            'std_mean_range': std_mean_range,
            'sample_update_interval': sample_update_interval,
            'sample_update_interval': True,
        },
        inference_only=inference_only,
    )

def log_random_memory_policy(
    action_update_interval: int,
    std_low: float,
    std_high: float,
    mean_mean: float = 0.0,
    sample_update_interval: bool = False,
    inference_only: bool = False,
) -> Dict[str, Any]:
    """
    Config for random exploration policy ``LogMultiRandomMemoryPolicy``
    """
    return _expl_policy(
        LogMultiRandomMemoryPolicy,
        {
            'action_update_interval': action_update_interval,
            'mean_mean': mean_mean,
            'std_low': std_low,
            'std_high': std_high,
            'sample_update_interval': sample_update_interval,
        },
        inference_only=inference_only,
    )