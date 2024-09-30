# Policies

This submodule defines abstract base classes for policies and value functions (*base.py*) and provides a few instantiations for them (*meta_policy.py* and *meta_value_function.py*). Additionally, random policies are instantiated in *exploration.py*.

## Base classes for policies and value functions
Each policy (base class ``Policy``) is derived from ``torch.nn.Module``. It implements a function which returns the distribution over actions (``torch.Tensor``), computed from arbitrary inputs: ``forward()``. It also implements functions which directly samples actions (``numpy.ndarray``) based on observations (``get_action()`` and ``get_actions()``).

Derived classes of ``Policy`` refine the type of inputs which the policy expects to observations (``StandarPolicy``) or observations and tasks (``MultiTaskPolicy``).
The class ``MetaRLPolicy`` is very similar but represents that task representations may not convey ideal information about the true task.

Similarly to the policies, value functions (state-value or state-action-value) are defined by means of very abstract base classes (``ValueFunction``) and specified sub-versions of these (e.g. ``MultiTaskQFunction``, ``MetaQFunction``).

You can find all of the above classes in *base.py*. To instantiate them, derived classes mostly need to find a suitable implementation for the ``forward()`` function.

## Instantiations of policies and Q-functions
The files *meta_policy.py* and *meta_value_function.py* provide simple implementations of ``MetaRLPolicy`` and ``MetaQFunction`` which are based on multilayer perceptron architectures.

## Exploration policies
The file *exploration.py* implements random policies for exploration, including:
- ``RandomPolicy``: A fully random policy (derived from ``StandardPolicy``) which samples actions from a Gaussian distribution
    $$ a \sim \mathcal{N}(\mu, \sigma^2 I) $$
- ``RandomMemoryPolicy``: A random policy which uses a memory to keep actions consistent over some number $k$ of timesteps. It samples the mean and standard deviation of ``RandomPolicy`` after $k$ steps:
    $$ \mu \sim \mathcal{N}(\hat{\mu}, \hat{\sigma}^2 I), \quad
		\sigma \sim \text{Exp}(\tilde{\sigma}) \qquad \text{every $k$ steps} $$
- ``MultiRandomMemoryPolicy``: Builds on top of ``RandomMemoryPolicy``. Every time that ``reset()`` is called, the mean and standard deviation of the policy change according to a distribution.
    $$ \hat{\mu} \sim \mathcal{N}(m, sI), \quad \hat{\sigma} \sim \text{Uniform} (\hat{\sigma}_{low}, \hat{\sigma}_{high}), \quad \tilde{\sigma} \sim \text{Uniform} (\tilde{\sigma}_{low}, \tilde{\sigma}_{high}) \\
		\text{at the beginning of each new episode} $$
- ``LogMultiRandomMemoryPolicy``: Similar to  ``MultiRandomMemoryPolicy`` but with slightly different sampling rule which stresses different scales of variation.
    $$ 	\hat{\mu} \sim \mathcal{N}(m, sI), \quad \log \hat{\sigma} \sim \text{Uniform} (\log \hat{\sigma}_{low}, \log \hat{\sigma}_{high}), \quad \tilde{\sigma} = \hat{\sigma} \\
		\text{at the beginning of each new episode} $$