"""
This module contains equivariant versions of GRU cells and GRU networks.

Author(s): 
    Julius Durmann, adapted from Jonas Jürß!
Contact: 
    julius.durmann@tum.de
Date: 
    2023-01-13

Reference:
    Jonas Jürß, SAM-RL
"""

import torch
import numpy as np

from symmetrizer.nn.modules import BasisLinear
from symmetrizer.groups import Group

import rlkit.torch.pytorch_util as ptu


class EquivariantGRUCell(torch.nn.Module):
    """An equivariant GRU network

    Parameters
    ----------
    input_length : int
        Input channels
    hidden_length : int
        Hidden and output channels
    input_group : symmetrizer.groups.Group
        Equivariance group for inputs x
    hidden_group : symmetrizer.groups.Group
        Equivariance group for hidden variables h
    """
    def __init__(self, input_group: Group, hidden_group: Group, channels_in: int = 1, hidden_channels: int = 1):
        super(EquivariantGRUCell, self).__init__()
        self.input_length = channels_in
        self.hidden_length = hidden_channels

        # Reset gate
        self.reset_gate_x = BasisLinear(channels_in, hidden_channels, input_group, bias=True)
        self.reset_gate_h = BasisLinear(hidden_channels, hidden_channels, hidden_group, bias=True)

        # Update gate
        self.update_gate_x = BasisLinear(channels_in, hidden_channels, input_group, bias=True)
        self.update_gate_h = BasisLinear(hidden_channels, hidden_channels, hidden_group, bias=True)

        # Candidate activation
        self.candidate_gate_x = BasisLinear(channels_in, hidden_channels, input_group, bias=True)
        self.candidate_gate_h = BasisLinear(hidden_channels, hidden_channels, hidden_group, bias=True)

        self.reset_activation = torch.nn.Sigmoid()
        self.update_activation = torch.nn.Sigmoid()
        self.candidate_activation = torch.nn.Tanh()

        self.input_size = self.reset_gate_x.repr_size_in
        self.output_size = self.candidate_gate_x.repr_size_out

    def reset_gate(self, x, h):
        x_1 = self.reset_gate_x(x)
        h_1 = self.reset_gate_h(h)
        # gate update
        reset = self.reset_activation(x_1 + h_1)
        return reset

    def update_gate(self, x, h):
        x_2 = self.update_gate_x(x)
        h_2 = self.update_gate_h(h)
        z = self.update_activation(h_2 + x_2)
        return z

    def candidate_component(self, x, h, r):
        x_3 = self.candidate_gate_x(x)
        h_3 = r * self.candidate_gate_h(h)
        gate_update = self.candidate_activation(x_3 + h_3)
        return gate_update

    def forward(self, x, h):
        # Equation 1. reset gate vector
        r = self.reset_gate(x, h)

        # Equation 2: the update gate - the shared update gate vector z
        z = self.update_gate(x, h)

        # Equation 3: The almost output component
        n = self.candidate_component(x, h, r)

        # Equation 4: the new hidden state
        h_new = (1-z) * n + z * h

        return h_new

class EquivariantGRU(torch.nn.Module):
    """
    An equivariant GRU cell.

    NOTE: Not all equivariant groups are compatible with gated recurrent units!
    Be especially aware of their element-wise computations!

    ---------------------------------------------------------------------------
    Inputs have shape ``(<batch_dims>, sequence_length, channels_in, input_dim)``
        OR ``(sequence_length, <batch_dims>, channels_in, input_dim)``
    Outputs will have shape ``(<batch_dims>, sequence_length, channels_out, input_dim)``
        OR ``(sequence_length, <batch_dims>, channels_out, input_dim)``
    You can use the parameter ``batch_first`` to determine order of batch dimensions
    and sequence dimension.
    ---------------------------------------------------------------------------

    Note: The equivariances of hidden_group should be the same as the output 
    transformation of the input group.

    Parameters
    ----------
    input_group : Group
        Equivariance group for inputs
    hidden_group : Group
        Equivariance group for hidden states
    channels_in : int, optional
        Number of input channels, allows multiple weigths although input and
        output size are bound by input_group!
        By default 1
    channels_out : int, optional
        Number of output channels, allows multiple weigths although input and
        output size are bound by input_group!
        By default 1
    batch_first : bool, optional
        Set to ``True`` to have batch dimensions before sequence dimension.
    """
    def __init__(self, input_group: Group, hidden_group: Group, channels_in: int = 1, channels_out: int = 1, batch_first: bool = False):
        super(EquivariantGRU, self).__init__()
        self.input_group = input_group
        self.cell = EquivariantGRUCell(input_group, hidden_group, channels_in=channels_in, hidden_channels=channels_out)

        self._batch_first = batch_first

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.input_size = self.cell.input_size
        self.output_size = self.cell.output_size

    def forward(self, input: torch.Tensor, h0: torch.Tensor = None):

        # Input has shape (<batch_dims>, sequence_length, channels_in, input_dim)
        # OR (sequence_length, <batch_dims>, channels_in, input_dim)

        # Output will have shape (<batch_dims>, sequence_length, channels_out, input_dim)
        # OR (sequence_length, <batch_dims>, channels_out, input_dim)

        # Dimension checking
        n_batch_dims = input.ndim - 3
        assert n_batch_dims >= 0, "Input must have at least three dimensions!"
        unbatched = n_batch_dims == 0
        if self._batch_first:   # Make sure that sequence dimension is at position 0
            input = input.permute([n_batch_dims, *range(n_batch_dims), -2, -1])
        if unbatched:
            input = input.unsqueeze(1)  # Make sure that input is batched
            n_batch_dims = 1
        batch_dims = input.shape[1:-2]  # (batch_dim_1, ..., batch_dim_n)

        # Input tensor has shape (sequence_length, batch_dim_1, ..., batch_dim_n, channels_in, input_size)
        sequence_length, channels_in, input_size = input.shape[0], input.shape[-2], input.shape[-1]
        assert channels_in == self.channels_in, f"Input must have {self.channels_in} input channels."

        # (Reduce multiple batch dimensions to one if necessary)
        if n_batch_dims > 1:
            input = input.reshape([sequence_length, np.prod(batch_dims), self.channels_in, input_size])

        # Initial hidden state
        if h0 is not None:
            h_t = h0
        else:
            h_t = torch.zeros((np.prod(batch_dims), self.channels_out, self.output_size), device=input.device)

        # >> Pass sequence through GRU cell <<
        h = torch.zeros((sequence_length, np.prod(batch_dims), self.channels_out, self.output_size), device=input.device)
        for t, x_t in enumerate(input):
            h_t = self.cell(x_t, h_t)
            h[t] = h_t

        # Remove batch dim if inputs were unbatched
        if unbatched:
            h = h.squeeze(1)
            n_batch_dims = 0
            batch_dims = []
            h_t = h_t.squeeze(0)
            
        # Regain batch dimensions
        h = h.reshape([sequence_length, *batch_dims, self.channels_out, self.output_size])
        if self._batch_first: # (Change back to batch-dimension-first if necessary)
            h = h.permute([*range(1, n_batch_dims+1), 0, -2, -1])


        return h, h_t

