import traceback
import torch
import numpy as np
from collections.abc import Mapping, Sequence, Iterator
from collections import deque
from typing import List, Union, Dict, Tuple

array_like = Union[np.ndarray, torch.Tensor]


class TransitionSequence(Mapping, Sequence):
    """A container class for transition sequences.

    This class holds values for
    - observations
    - actions
    - rewards
    - next observations
    - terminals
    - user-specific values (use ``additional_items`` argument)

    It functions as a dictionary when used with string keys and as a list
    when used with integer indices. The dictionary differentiates between
    value types (observations, actions, etc.) while the list differentiates
    between timesteps.

    NOTE: This class is not ready to hold batched data!

    Indexing is supported in two different ways:

    String keys:
    ```
    sequence['observations']
    ```
    returns the observations as numpy array or torch tensor.
    You can alternatively call
    ```
    sequence.observations
    ```

    Integer indices:
    ```
    sequence[1]
    sequence[1:4]
    sequence[1,2,4]
    ```
    This returns a new ``TransitionSequence`` object with the reduced number
    of steps.
    Likewise, you can use integer indices to assign values to the sequence.

    You should be able to use most of the list comprehension tools from python, 
    including
    ```
    sequence += [(<new_obs>, <new_act>, ..., <new_term>), ...]
    (step for step in sequence)
    ```
    """
    def __init__(
        self, 
        maxlen: int = None, 
        additional_items: List[str] = None, 
        observations: array_like = None, 
        actions: array_like = None, 
        rewards: array_like = None, 
        next_observations: array_like = None, 
        terminals: array_like = None,
        **others: array_like,
    ) -> None:
        self.maxlen = maxlen

        if additional_items is None:
            additional_items = []
        
        self._keys: List[str] = ['observations', 'actions', 'rewards', 'next_observations', 'terminals'] + additional_items
        self._additional_keys = additional_items
        self._data: Dict[str, List[array_like]] = {k: deque([], maxlen=maxlen) for k in self._keys}

        if observations is not None:
            for step in observations.shape[0]:
                self.add_step(
                    observations[step], 
                    actions[step],
                    rewards[step],
                    next_observations[step],
                    terminals[step],
                    **{k: value[step] for k, value in others.items()},
                )

        # OPTION: Adapt code to this container class.

    def get_step(self, idx: int) -> Tuple[array_like, ...]:
        """Get values from transition at ``idx``. You can also use indexing.

        Parameters
        ----------
        idx : int
            Index for transition

        Returns
        -------
        Tuple[array_like, ...]
            Observations, actions, rewards, next observations, terminals, ...
        """
        step = []
        for value in self._data.values():
            step.append(value[idx])
        return tuple(step)

    def _set_step(self, idx: int, **__value_dict):
        for key in self._data.keys():
            self._data[key][idx] = __value_dict[key]

    def _set_steps(self, idcs: Iterator, values: Tuple[Tuple[array_like, ...], ...]):
        if not isinstance(values[0], tuple):
            values = [values]
        for i, idx in enumerate(idcs):
            v = values[i] if len(values) > 1 else values[0]
            self.set_step(idx, *v) 

    def set_step(self, idx: int, \
        obs: array_like, act: array_like, rew: array_like, \
        next_obs: array_like, terminal: array_like, \
        *others: array_like, **others_: array_like
    ):
        """Set values of a single step. You can also use indexing.

        Parameters
        ----------
        idx : int
            Index of the transition which should be changed
        *values : array_like
            Values for this transition
        """
        input = {
            'observations': obs,
            'actions': act,
            'rewards': rew,
            'next_observations': next_obs,
            'terminals': terminal,
            **{self._additional_keys[i]: value for i, value in enumerate(others)},
            **others_,
        }
        self._set_step(idx, **input)

    def add_step(self, obs: array_like, act: array_like, rew: array_like, next_obs: array_like, terminal: array_like, *others: array_like, **others_: array_like):
        """Add a single step. You can also use ``append()``.

        Parameters
        ----------
        *values : array_like
            Values for this transition
        """
        input = {
            'observations': obs,
            'actions': act,
            'rewards': rew,
            'next_observations': next_obs,
            'terminals': terminal,
            **{self._additional_keys[i]: value for i, value in enumerate(others)},
            **others_,
        }
        for key, value in input.items():
            self._data[key].append(value)

    def append(self, obs: array_like, act: array_like, rew: array_like, next_obs: array_like, terminal: array_like, *others: array_like, **others_: array_like):
        """Same as ``add_step()``.
        """
        self.add_step(obs, act, rew, next_obs, terminal, *others, **others_)

    def extend(self, steps: List[Tuple[np.ndarray, ...]]):
        """Extend the list of transitions by values

        Parameters
        ----------
        steps : List[Tuple[np.ndarray, ...]]
            List of transition tuples
        """
        for step in steps:
            self.append(*step)

    
    # vvv Indexing support and convenience functions... vvv

    def __add__(self, steps: List[Tuple[np.ndarray, ...]]):
        self.extend(steps)
        return self

    def __getattr__(self, name: str):
        if name == "_data":
            raise AttributeError
        try:
            attribute = self._data[name]
            if isinstance(attribute[0], np.ndarray):
                return np.array(attribute)
            elif isinstance(attribute[0], torch.Tensor):
                return torch.concatenate(tuple(attribute), dim=0)
            else:
                raise RuntimeError("Stored attributes are not numpy array or torch tensors!")
        except KeyError:
            traceback.print_exc()
            raise KeyError(f"This TrajectorySequence object has no attribute '{name}'")

    def __getitem__(self, key: Union[str, int, slice, Tuple[int, ...]]) -> Union[array_like, Tuple[array_like, ...]]:
        if isinstance(key, str):
            return getattr(self, key)
        elif isinstance(key, int):
            return self.get_step(key)
        elif isinstance(key, slice):
            sequence = TransitionSequence(additional_items=self._additional_keys)
            for i in range(*key.indices(len(self))):
                sequence.append(*self.get_step(i))
            return sequence
        elif isinstance(key, tuple):
            sequence = TransitionSequence(additional_items=self._additional_keys)
            for i in key:
                sequence.append(*self.get_step(i))
            return sequence
        elif isinstance(key, list):
            return self.__getitem__(tuple(key))
        else:
            raise ValueError("Invalid indices or keys!")

    def __setitem__(self, key: Union[str, int, slice, Tuple[int, ...]], *values: Union[array_like, Tuple[array_like, ...], Tuple[Tuple[array_like, ...], ...]]):
        if isinstance(key, str):
            assert len(values) == 1, f"Can only accept one input for '{key}'"
            assert isinstance(values[0], np.ndarray) or isinstance(values[0], torch.Tensor), "Assigned value must be of type numpy.ndarray or torch.Tensor!"
            self._data[key] = values[0]
        elif isinstance(key, int):
            return self.set_step(key, *values[0])
        elif isinstance(key, slice):
            self._set_steps(range(*key.indices(len(self))), *values)
        elif isinstance(key, tuple):
            self._set_steps(key, *values)
        elif isinstance(key, list):
            self._set_steps(tuple(key), *values)
        else:
            raise ValueError("Invalid indices or keys!")

    def __len__(self) -> int:
        return len(self._data['observations'])

    def __iter__(self) -> Iterator[Tuple[np.ndarray, ...]]:
        for i in range(self.__len__()):
            yield self.get_step(i)

    def keys(self) -> List[str]:
        return self._keys

    def items(self):
        values = (getattr(self, key) for key in self._keys)
        return self._keys, values

    def __str__(self) -> str:
        s = f"{'Step':<10}\t"
        for key in self._keys:
            s += f"{key:<10}\t"
        for i, step in enumerate(range(len(self))):
            s += "\n"
            s += f"{i:<10}\t"
            for key in self._keys:
                s += f"{self._data[key][step]}\t"
        return s

    def __repr__(self) -> str:
        s = f"Transition sequence with keys "
        for key in self._keys:
            s += f"'{key}', "
        s += f"and length {len(self)}"
        return s


if __name__ == "__main__":
    s = TransitionSequence(maxlen=15, additional_items=["encodings"])

    observations = np.random.randn(10, 3)
    actions = np.random.randn(10, 2)
    rewards = np.random.randn(10, 1)
    next_observations = np.random.randn(*observations.shape)
    terminals = np.random.randint(0, 1, [10, 1])
    encodings = np.random.randn(10, 3)

    for obs, act, rew, next_obs, terminal, encoding in zip(observations, actions, rewards, next_observations, terminals, encodings):
        s.add_step(obs, act, rew, next_obs, terminal, encoding)

    print(s)
    s[0] = np.array([0,0,0]), np.array([3,3]), np.array([-1]), np.array([3,4,5]), np.array([0]), np.array([0, 0, 0])
    s[1:3] = (np.array([1,2,3]), np.array([3,3]), np.array([-1]), np.array([3,4,5]), np.array([0]), np.array([0, 0, 0])), (np.array([8,9,10]), np.array([3,3]), np.array([-1]), np.array([3,4,5]), np.array([0]), np.array([0, 0, 0]))
    s[6:8] = (np.array([1,2,3]), np.array([3,3]), np.array([-1]), np.array([3,4,5]), np.array([0]), np.array([0, 0, 0]))
    print(s)


    s += zip(observations, actions, rewards, next_observations, terminals, encodings)
    print(s)

    print(s.observations)

    for step in s:
        print(step)