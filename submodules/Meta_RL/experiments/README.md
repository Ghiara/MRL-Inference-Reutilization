# Experiments

In this folder, you can find 

- Experiment configurations (*experiments*)
- Modules for experiment configurations (*config_modules*)
- Functions for experiment analysis (*analysis*)

## Experiment configurations
Experiment configurations are dictionaries which can be passed to *smrl > experiment > experiment_setup.py > ``setup_experiment()``*. They contain information about network configurations, training utility, and training parameters.

## Configuration modules
Configuration modules define functions which return parts of a configuration dictionary. They are mostly used to avoid redundant code and to simplify the experiment configurations.

## Analysis functions
Analysis functions can be used to evaluate training progress and the final performance of trained agents. 

Most relevant is the function *model_evaluation.py > ``model_evaluation()``*. It visualizes important training curves and rolls out trajectories for creating plots which show the latent space of the encoder and movements of the agent.