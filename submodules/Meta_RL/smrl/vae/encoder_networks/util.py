import torch
from typing import Callable, Tuple, Union


def batched(fn: Callable):
    """
    Ensure that context encoder inputs are batched, i.e. that they have shape
    (batch_size, sequence_length, *).
    """
    def forward_fn(*args, **kwargs):
        batched_args = []
        batched_kwargs = {}
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.ndim < 2:
                raise ValueError("Inputs cannot have less than two dimensions,"
                    + " they must either have shape (sequence_length, *) or"
                    + " (batch_size, sequence_length, *)"
                )
            if isinstance(arg, torch.Tensor) and arg.ndim == 2: 
                arg = arg[None, ...]
            batched_args.append(arg)
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.ndim < 2:
                raise ValueError("Inputs cannot have less than two dimensions,"
                    + " they must either have shape (sequence_length, *) or"
                    + " (batch_size, sequence_length, *)"
                )
            if isinstance(value, torch.Tensor) and value.ndim == 2: 
                value = value[None, ...]
            batched_kwargs[key] = value
        return fn(*batched_args, **batched_kwargs)
    return forward_fn

def pad_sequence_length(sequence_length: int, *args: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    """
    Ensure that context encoder inputs have the correct sequence length.

    This function pads the second dimension of input tensors with shape
    (batch_dim, some_other_sequence_length, *).

    Parameters
    ----------
    sequence_length : int
        Desired sequence length
    *args : torch.Tensor(s)
        Tensors with shape (batch_dim, some_other_sequence_length, *)

    Returns
    -------
    torch.Tensor(s)
        Tensors with shape (batch_dim, sequence_length, *)
    """
    padded_args = []
    for x in args:
        l = x.shape[1]
        if x.shape[1] < sequence_length:
            x = torch.nn.functional.pad(x, (0,0, sequence_length - x.shape[1], 0, 0, 0))
        if x.shape[1] > sequence_length:
            x = x[:, 1:, :] # Extended context, probably due to additional information in ELBO or SAC losses
        if x.shape[1] != sequence_length:
            raise ValueError(
                "Inputs do not have the right sequence length "
                + f"(should be less than or equal to {sequence_length} "
                + f"but was {l})!"
            )
        padded_args.append(x)
    if len(padded_args) == 1:
        return padded_args[0]
    else:
        return tuple(padded_args)

def at_least_one_timestep(*args: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    padded_args = []
    for arg in args:
        assert arg.ndim >= 3, "Argument must have at least three dimensions: (batch_size, sequence_length, *)"
        if arg.shape[1] < 1:
            arg = pad_sequence_length(1, arg)
        padded_args.append(arg)
    if len(padded_args) == 1:
        return padded_args[0]
    else:
        return tuple(padded_args)