"""
This module contains transformation invariance groups.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-02-02
"""

import numpy as np

from symmetrizer.nn.modules import BasisLinear
from symmetrizer.ops import GroupRepresentations
from symmetrizer.groups import Group

permutation_repr = GroupRepresentations(
    [
        np.eye(2),
        np.array([[0, 1], [1, 0]])
    ],
    name = "Permutation",
)


class PermutationGroup(Group):
    """
    Equivariance group of 2d permutations.
    """
    def __init__(self):
        self.parameters = range(2)

        self._transforms = permutation_repr

        self.repr_size_in = 2
        self.repr_size_out = 2
    
    def _input_transformation(self, weights, idx: int):
        return weights @ self._transforms[idx]

    def _output_transformation(self, weights, idx: int):
        return self._transforms[idx] @ weights

class PermutationToNegationGroup(Group):
    """
    Equivariance group from 2d permuation transformations to negation transformations.
    """
    def __init__(self, repr_size_out: int):
        self.parameters = range(2)
        self._input_transforms = permutation_repr
        self._output_transforms = [1.0, -1.0]
        self.repr_size_in = 2
        self.repr_size_out = repr_size_out

    def _input_transformation(self, weights, idx: int):
        return weights @ self._input_transforms[idx]

    def _output_transformation(self, weights, idx: int):
        return self._output_transforms[idx] * weights

class PermutationToInvariantGroup(Group):
    """
    Equivariance group from 2d permuation transformations to identity transformations.

    Parameters
    ----------
    Group : _type_
        _description_
    """
    def __init__(self, repr_size_out: int):
        self.parameters = range(2)
        self._input_transforms = permutation_repr
        self._output_transforms = [lambda x: x, lambda x: x]
        self.repr_size_in = 2
        self.repr_size_out = repr_size_out

    def _input_transformation(self, weights, idx: int):
        return weights @ self._input_transforms[idx]

    def _output_transformation(self, weights, idx: int):
        return self._output_transforms[idx](weights)