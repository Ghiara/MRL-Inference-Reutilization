# Inference-based Meta-Reinforcement-Learning
Meta-Reinforcement Learning with an encoder module which infers tasks from contexts.

This project builds upon [RLKIT][RLKIT].

In this project, we implement methods to train an agent which can act reasonably in meta-environments, i.e. environments which may have varying transition dynamics or reward functions - described by **tasks**. For this, the agent adapts to the current task by inferring a compact representation of it from few recent transition samples. This lets it modify its behavior within a small number of transitions.

This project is related to previous work, including work by Kate Rakelly et al. [1], David Lerch [2], Lukas Knak [3], Philipp Widmann [4], and Jonas Jürß [5].

For more information about the theoretical background, please refer to my Master's thesis [8] and to the references.


## Contents
1. [Installation](#installation)
    1. [Cloning](#cloning)
    2. [Environment](#environment)
    3. [Submodules](#submodules)
    4. [Locally install SMRL package](#locally-install-smrl-package)
2. [Run code](#run-code)
    1. [Easy start: sample script *run_experiment.py*](#easy-start-sample-script-run_experimentpy)
    2. [Configurations](#configurations)
    3. [Environment factory functions](#environment-factory-functions)
    4. [Function *setup_experiment*](#function-setup_experiment)
    5. [Multithreading](#multithreading)
    6. [Restrict GPU useage](#restrict-gpu-usage)
3. [Runner script (console interface)](#runner-script-console-interface)
4. [Code](#code)
5. [Docker](#docker)
    1. [Installation](#docker-installation)
    2. [Docker image](#create-docker-image)
    3. [Run container](#run-container-interactively)
6. [Testing environment](#testing-environment)
7. [Known issues](#known-issues)
8. [References](#references)

----------------------------------------------------------------------------

## Installation
### Cloning
Go to the GitHub repository website and select 'Code' to get an HTTPS or SSH link to the repository.
Clone the repository to your device, e.g.
```bash
git clone git@github.com:juldur1/Symmetric-Meta-Reinforcement-Learning.git
```
Enter the root directory of this project on your device. The root directory contains this README-file.

### Environment

We recommend to manage the python environment with **conda** and suggest [Miniconda](https://docs.conda.io/en/latest/miniconda.html) as a light-weight installation.

> You may also use different environment tools such as python's *venv*. Please refer
to *requirements.txt* in this case. In the following, we will proceed with conda.

> If you prefer to use *Docker*, skip the following steps and proceed with [this section](#docker).

Install the environment using the ``conda`` command:
```bash
conda env create --file environment.yml
```
This might take some time because it needs to download and install all required packages.
> 💡 **NOTE**: In case you want to use GPU-accelerated training on NVIDIA-GPUs, *uncomment* the following line in *environment.yml* before running `conda env create`:
> ```yml
> #- pytorch-cuda=11.7 # Consider 'cpuonly' instead if you are only running on a CPU
> ```
> and *comment out* (or delete) the following line to make sure pytorch is set up to run with CUDA:
> ```yml
> - cpuonly
> ```

Activate the new environment by running (Make sure that no other environment was active before.):
```bash
conda activate SMRL
```

In case you need to update an existing environment, you can run
```bash
conda env update --file environment.yml
```

### Submodules
Initialize the submodules and download their code.
```bash
git submodule init
git submodule update
```

Install the submodules:
```bash
pip install -e ./submodules/meta-environments
pip install -e ./submodules/MRL-analysis-tools
pip install -e ./submodules/rlkit
pip install -e ./submodules/symmetrizer     
```
You only need the last submodule if you want to use equivariant networks.

### Locally install SMRL package
Similar to the submodules, the SMRL package also needs to be installed locally.
This step ensures that the package can be referenced as any other python package.

Please run (from the root directory):
```batch
pip install -e .
```

By running `conda list` you can check if all packages have been installed successfully.

### MuJoCo

MuJoCo is only required if you want to run MuJoCo environments such as Half-Cheetah and Ant. If you do not want to use them, make sure to comment or delete MuJoCo references in *configs/environment_factory.py* (and some other files...).

For running MuJoCo environments, you will need the MuJoCo library binaries which you can download from [here](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz). You should extract them in the directory */root/.mujoco/mujoco210*. Additionally, you need to set the variable *LD_LIBRARY_PATH* to contain the path to the binaries:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
```

Please read the [documentation](https://github.com/openai) *carefully* (It contains valuable hints on troubleshooting).

The python bindings for MuJoCo can be installed with

```bash
pip install mujoco-py
```

----------------------------------------------------------------------------

## Run code

### Easy start: sample script *run_experiment.py*
The script *run_experiment.py* provides a small working example. 
It ...

1. ... imports a [configuration dictionary](#configurations)
2. ... sets the environment factory function
3. ... checks for GPU availability and initializes multithreading
4. ... sets logger arguments
5. ... instantiates all networks and the training algorithm (see [here](#function-setup_experiment))
6. ... starts training

You can modify the code to start your own experiments.

### Configurations
Configuration dictionaries serve as a tool to describe experiments.
They ...

1. ... determine which networks are used
2. ... determine which inference mechanism is used
3. ... set arguments for networks
4. ... set arguments for trainers and the algorithm
5. ... etc.

These dictionaries must contain some required key-value pairs and may contain optional key-value pairs. You can find a definition of the dictionary layout in [*smrl/experiment/experiment_setup.py*](smrl/experiment/experiment_setup.py) (see for dictionary ``base_config``). 

For convenience, you can create a python file with the dictionary. Feel free to have a look at the provided configuration files (which contain dictionaries) in *./configs/*.

It is worth mentioning that many of the entries in the dictionaries are *types* (classes). This hopefully makes them easier to read and allows to get hints to the classes in many code editors.

### Environment factory functions
Environment factory functions are a part of configuration dictionaries. They tell the algorithm setup functionality which environments are used for exploration during training and evaluation. The must follow the signature
```python
def environment_factory_function() -> Tuple[MetaEnv, MetaEnv]:
    # ...
    return expl_env, eval_env
```
For an example of such functions, refer to [*configs/environment_factory.py*](configs/environment_factory.py).

### Function ``setup_experiment()``
The function ``setup_experiment()`` (located in [*smrl/experiment/experiment_setup.py*](smrl/experiment/experiment_setup.py)) sets up all networks and mechanisms which are required for training:
1. Networks for SAC-training (policy and value functions)
2. Networks for inference training (encoder and decoder)
3. Rollout utility and data buffers
4. Trainers for policy (SAC) and inference mechanism
5. Algorithm

It returns an instance of ``MetaRlAlgorithm`` which can be directly used for training.

### Multithreading
The SMRL package supports multithreaded trajectory rollouts with `ray` ([ray documentation](https://docs.ray.io/en/latest/index.html)).
To use this functionality, include
```python
import os
os.environ["MULTITHREADING"] = "True"
```
before calling ``setup_experiment()``. The setup utility then automatically initializes ray and all components of the code know about multithreading.

### Restrict GPU usage
In case you have multiple GPUs available and only use some of them, you can add the following to the top of your python script:
```python
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="<device id, e.g. 0>"
```
The first command ensures that the internal listing of GPUs is the same as with the ``nvidia-smi`` command. To list available GPUs and their current load, you can run
```bash
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory --format=csv
```

## Runner script (console interface)
There is also a runner script which can be called from a console. You can start training with the following command:
```bash
python runner.py <environment> <configuration>
```
The argument ``<environment>`` should point towards an environment factory function while ``<configuration>`` declares which configuration dictionary to use. Both arguments should be passed in **python import notation**, e.g. ``"configs.base_configuration.config"``. They will be parsed internally and the referenced python object will be imported. Additional parameters exist as well which allow to select directories for storing results, managing gpu usage, etc. 

For more information, use
```bash
python runner.py --help
```

----------------------------------------------------------------------------

## Code
<!-- tree ./smrl -d -L 1 -->
```
./smrl
├── algorithms
├── data_management
├── environments
├── experiment
├── policies
├── trainers
├── utility
└── vae
```

For information about the code, please refer to the readme-file in folder *smrl*.


----------------------------------------------------------------------------

## Docker
[Docker](https://www.docker.com/) provides utilities to run applications in separate *containers*. If you are mainly interested in *running* the code and *code modification* is *not* your primary interest, you may prefer this installation method. The steps below will guide you towards creating a Docker container which can run the code from this package.

### Docker installation
Please visit the [Docker website](https://www.docker.com/) and follow the installation instructions.

### Submodules
Initialize the submodules and download their code such that they can be copied to the docker image in the next step.
```bash
git submodule init
git submodule update
```

### Create Docker image
A container is a running instance of an image. With the help of the *Dockerfile*, you can create your own image. 

> NOTE: In order to use GPU-accelerated training, follow the instructions in *requirements.txt*!

To do so, run

```bash
docker build -t smrl .
```

NOTE: This step may take some time as it sets up an operating system and a python environment, including all the required packages.

You can check if the image has been created successfully by running
```bash
docker image ls
```

To check the installation, run the image by calling
```bash
docker run smrl
```
This should trigger *check_setup.py* which tests if all required packages are successfully installed.


### Run container interactively
Create a directory for data, name *data/* in the projects root directory. This directory will be mounted by the running containers at */data/* such that data is always available. 

After the image has been created, you can run an interactive container by executing
```bash
docker run -i -t --mount type=bind,source="$(pwd)"/data,target=/data smrl bash
```

> NOTE: If you are on Windows, replace ``$(pwd)`` by ``%cd%``!

You should be prompted with a linux console in directory */home/Symmetric-Meta-Reinforcement-Learning*. From this console, you can execute any python script by calling

```bash
python <script.py>
```

Remember to set the storage directory to */data/* if you want to have persistent data which is also available at the host system. In interactive mode, you may also copy any data to or from this directory in order to access information from the host system or provide results.

> NOTE: The Docker image is not updated automatically when you modify anything in the package directory. You will need to recreate the image before changes are applied. Also be aware that any changes within the Docker container will not be available once the container is shut down. 

You can keep track over running and stopped containers by executing
```bash
docker container ls -a
```

> NOTE: Every time you run ``docker run`` you create and start a new container from the image. Over time, there thus may be many unused containers of the *smrl* image. Consider deleting them as you like.


----------------------------------------------------------------------------

## Testing environment
- Windows Subsystem for Linux:
    - Windows 10 Home, Version 22H2 (host system)
    - Ubuntu Ubuntu 20.04.5 LTS
    -   | System specifications |                               |
        | --------- | ---------------------------------------   |
        | Processor | Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz  |
        | RAM       |   8 GB                                    |
        | GPU       | -                                         |
- Ubuntu (server):
    - Ubuntu 18.04.3 LTS
    -   | System specifications |                               |
        | --------- | ---------------------------------------   |
        | Processor | Intel(R) Xeon(R) Gold 6134 CPU @ 3.20GHz  |
        | RAM       |   252 GB                                  |
        | GPU       | NVIDIA Tesla V100-PCIE-32GB               |

----------------------------------------------------------------------------

## Known issues

### Multithreading incompatible with symmetrizer networks
Symmetrizer networks are not compatible with multithreading. An error which states
that objects cannot be deserialized to CUDA is raised.

### Multithreading incompatible / inefficient with MuJoCo simulations
Using multithreading with ``ray`` when the environment is simulated by MuJoCo
is slow and inefficient. Over time, the memory usage grows. We recommend disabling
multithreading (see [Multithreading](#multithreading)) since MuJoCo seems to have
its internal hardware management.

----------------------------------------------------------------------------

## References

[1] Rakelly, K. et al. (2019) ‘Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables’, Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA: PMLR, pp. 5331–5340. Available at:
http://proceedings.mlr.press/v97/rakelly19a.html.

[2] Lerch, D. (2020) Meta-Reinforcement Learning in Non-Stationary and Dynamic Environments. Master's thesis. Technical University of Munich.

[3] Knak, L. (2021) Task Inference Based Meta-Reinforcement Learning for Robotics Environments. Master's thesis. Technical University of Munich.

[4] Widmann, P. (2022) Task Inference for Meta-Reinforcement Learning in Broad and Non-Parametric Environments. Master's thesis. Technical University of Munich.

[5] Jürß, J. (2022) Exploiting Symmetries in Context-Based Meta-Reinforcement Learning. Bachelor's thesis. Technical University of Munich.

[6] RLKIT:  https://github.com/rail-berkeley/rlkit

[7] Symmetrizer:  https://github.com/ElisevanderPol/symmetrizer/

[8] Durmann, J. (2023) Meta-Reinforcement Learning. Master's thesis. Technical University of Munich.


<!--This part is for links only and won't be displayed in Markdown previews-->
[RLKIT]: <https://github.com/rail-berkeley/rlkit> "RLKIT on GitHub"

[Symmetrizer]: <https://github.com/ElisevanderPol/symmetrizer/> "Symmetrizer on GitHub"