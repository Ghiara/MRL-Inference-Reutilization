from typing import Dict, Any, List

from smrl.policies.meta_policy import *
from smrl.policies.meta_value_function import *


def policy_networks(hidden_sizes: List[int] = None) -> Dict[str, Any]:
    """
    Qfunction networks (MlpValueFunction) and Policy network (MetaRLTanhGaussianPolicy).

    Parameters
    ----------
    hidden_sizes : List[int]
        Hidden sizes of the policy and q-function networks.
        By default [16, 16, 16]
    """
    if hidden_sizes is None:
        hidden_sizes = [16, 16, 16]
    return dict(
        qf_network_type = MlpValueFunction,
        qf_network_kwargs = {
            'hidden_sizes': hidden_sizes,
        },
        policy_type = MetaRLTanhGaussianPolicy,
        policy_kwargs = {
            'hidden_sizes': hidden_sizes,
        },
    )