import numpy as np

from rlkit.policies.base import SerializablePolicy


class RandomPolicy(SerializablePolicy):
    """
    Policy that always outputs zero.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs, deterministic=False):
        return self.action_space.sample(), {}
    def get_actions(self, obs, deterministic=False):
        actions = []
        for i in range(obs.shape[0]):
            actions.append(np.random.rand(self.action_space)*2 -1)
        return np.array(actions)