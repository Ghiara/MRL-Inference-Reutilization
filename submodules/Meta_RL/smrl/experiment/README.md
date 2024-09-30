# Experiment
The files in this folder provide utility functions for setting up experiments from configuration dictionaries or loading trained models for evaluation. Most important is the file *experiment_setup*. Below, you will find a summary of the most relevant functions.

## *experiment_setup.py > ``setup_experiment()``*

This function is used to create an instance of ``MetaRlAlgorithm`` from configuration dictionaries. The dictionary needs to follow the structure which is defined by ``base_config`` (see also *experiment_setup.py*). It describes the components, parameters, and design choices of the experiment. Many of the entries in the dictionary are classes which determine the type of the components used in the experiment. Based on this information, models will be instantiated, trainers and training tools will be set up, and the algorithm instance will be created.

The function relies on subordinate functions such as ``init_networks()`` and ``setup_algorithm()``.

## *model_setup.py > ``init_networks()``*

This function initializes untrained networks, including the encoder and decoder, policy and Q-function networks. It can also determine between exploration and evaluation policy and may also instantiate separate policies for inference exploration.

## *model_setup.py > ``load_params()``*

This function is strongly related to ``init_networks()``. It laods model parameters from parameter files and can be used to continue training or analyze trained models.

## *_algorithm_setup.py > ``setup_algorithm``*

This function sets up a training algorithm, including data samplers, data buffers, and trainers for the models. It returns an instance of ``MetaRlAlgorithm`` which can be directly used for training.

# Analysis

The file *analysis.py* offers functions which automatically instantiate models from folders with configuration dictionaries, experiment progress data, and parameters. The model parameters are loaded from the respective file. The resulting output includes the trained models and information about the training progress. It can be used to evaluate performance of training and to visualize its progress.