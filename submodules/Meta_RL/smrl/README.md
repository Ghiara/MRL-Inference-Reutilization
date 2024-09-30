# SMRL
This directory contains the SMRL package which can train meta-reinforcement learning agents by learning how to infer tasks.

Each meta-RL agent consists of two components: An **inference module** which produces representations of tasks and an **actor-critic module** which learns to navigate in the environment and how to solve the task which is encoded by the inference module.

## Contents

1. [Package structure](#package-structure)
2. [Parts of the code](#parts-of-the-code)
    1. [Main algorithm](#main-algorithm-metarlalgorithm)
    2. [Agent component: Inference module](#agent-component-inference-module)
    3. [Agent component: Policy and Q-function](#agent-component-policy-and-q-function)
    4. [Trainers](#trainers)
    5. [Data management](#data-management)
    6. [Experiment setup](#experiment-setup)
    7. [Utility](#utility)
3. [References](#references)

## Package structure
The package has the following structure:

```
.
├── README.md           (this file)
├── algorithms
├── data_management
├── environments
├── experiment
├── policies
├── trainers
├── utility
└── vae
```


## Parts of the code

### Main algorithm: ``MetaRlAlgorithm``

The directory *algorithms* contains the main algorithm ``MetaRlAlgorithm`` (*meta_rl_algorithm.py*) which organizes training into 

1. Data sampling and storing
2. Policy training
3. Inference training

For each of the above steps it relies on code components from the other algorithms, e.g. from *data_management* for sampling and storing new trajectories, or from *trainers* for training the respective parts of the model.

### Agent component: Inference module
The inference module consists of an *encoder* and a *decoder* which form the components of a Variational Autoencoder (VAE) [1] or Neural Process (NP) [2]. They are trained simultaneously on a variational lower bound, called *"Evidence Lower BOund"* (ELBO), to predict transitions in the Markov decision process (MDP). The encoder is feeded with a sequence of recent transitions, called *context*, which allows it to infer relevant information about the task. Only the encoder is required for inferring representations of the task which are passed on to the policy and Q-function networks.

The directory *vae* contains definitions of abstract encoder, decoder, and VAE/NP classes for MDPs(see *mdp_vae.py*) and instantiations thereof (*encoder_networks*, *decoder_networks.py*).

For more details, refer to the readme-file of directory *vae*.

### Agent component: Policy and Q-function
For training the agent how to act, we choose *"Soft Actor Critic"* (SAC) [3]. Additional to their usual inputs (current state, current action), the actor- (policy) and critic-networks (Q-function) are also provided with a representation of the current task which is obtained from the encoder network.

The directory *policies* contains definitions of abstract policy and Q-function classes in this setting (*base.py*). Additionally, it provides instantiations of these classes. Moreover, the file *exploration.py* instantiates a set of random policy classes which can be used for exploration in environments with simple dynamics.

For more details, refer to the readme-file of directory *policies*.

### Trainers
Trainers implement the actual training steps and update the inference module, Q-function networks, and policy weights. Two separate algorithm exist for inference training (*vae.py*, ``MdpVAETrainer``) and behavior training (*meta_sac.py*, ``MetaSACTrainer``), respectively. These trainers accept batches of training data and implement the training steps of the [algorithm](#main-algorithm-metarlalgorithm).

### Environments
Directory *environments* contains definitions of abstract environment classes. These include the class ``MetaEnv`` which the rest of the code is built on.

### Data management
Directory *data_management* contains a suite of tools for sampling trajectories (*rollout_functions.py*, *path_collector.py*) and storing them (*replay_buffers*).

### Experiment setup
Directory *experiment* contains functions which setup algorithm components and return an instance of ``MetaRlAlgorithm``. They use configuration dictionaries to pass options and select instantiations of the components described above.

### Utility
Directory *utility* contains tools which are reused several times within the code. They serve a variety of different purposes, including dictionary handling and torch utility (*ops.py*), GPU management (*device_context_manager.py*), and console output formatting (*console_strings.py*).

---

## References
[1] Kingma, D.P. and Welling, M. (2014) ‘Auto-Encoding Variational Bayes’, 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings. Available at: http://arxiv.org/abs/1312.6114v10

[2] Garnelo, M. et al. (2018) ‘Neural Processes’, in ICML Workshop on Theoretical Foundations and Applications of Deep Generative Models. Available at: https://arxiv.org/abs/1807.01622

[3] Haarnoja, T. et al. (2018) ‘Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor’, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018: PMLR, pp. 1856–1865. Available at: http://proceedings.mlr.press/v80/haarnoja18b.html.