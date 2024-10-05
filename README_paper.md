LD_LIBRARY_PATH
:/home/hbp/.mujoco/mujoco200/bin:/home/bing/.mujoco/mjpro150/bin:/home/bing/.mujoco/mujoco200/bin:/home/bing/.mujoco/mjpro131/bin:/home/hbp/.mujoco/mjpro150/bin:/home/hbp/.mujoco/mjpro131/bin:/home/hbp/.mujoco/mujoco200/bin


(yuqi) hbp@aklma-MS-7B24:/usr/lib$ ls | grep nvidia
libnvidia-gtk2.so.460.39
libnvidia-gtk3.so.460.39
nvidia


# Meta-Reinforcement Learning in Broad and Non-Parametric Environments

This is the official implementation of the algorithm `Task-Inference-based meta-rl algorithm using Gaussian mixture models and gated Recurrent units (TIGR)`.

The implementation is based on [rlkit](https://github.com/vitchyr/rlkit), [PEARL](https://github.com/katerakelly/oyster) and [rand_param_envs](https://github.com/dennisl88/rand_param_envs.git).


## Requirements

- To install locally, you will need to first install [MuJoCo](https://www.roboti.us/index.html).
For the task distributions in which the reward function varies (Cheetah, Ant), install MuJoCo200.
(following is deprecated and will be removed) For the task distributions where different tasks correspond to different model parameters (Walker and Hopper), MuJoCo131 is required.
Simply install it the same way as MuJoCo200.
- Set `LD_LIBRARY_PATH` to point to both the MuJoCo binaries (`/$HOME/.mujoco/mujoco200/bin`) as well as the gpu drivers (something like `/usr/lib/nvidia-390`, you can find your version by running `nvidia-smi`).
- For the remaining dependencies, we recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html). Use
Use the latest `yml` file (`conda env create -f tigr.yml`) to set a conda virtual machine, or install the packages via `pip install -r requirements.txt`.
Make sure the correct GPU driver is installed and you use a matching version of CUDA toolkit and torch.
We use torch 1.7.0 with cuda version 11 for our evaluations (`pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html`)
- The implementation has a dependency on [Meta-World](https://github.com/rlworkgroup/metaworld). If not installed automatically via the `requirements.txt`, this has to be cloned and installed in the environment with
`git clone https://github.com/rlworkgroup/metaworld.git`
`cd metaworld`
`pip install -e .`
- We created versions of the standard Mujoco environments in the folder `submodules/meta_rand_envs`.
To perform experiments, both submodules `meta_rand_envs` and [rand_param_envs](https://github.com/dennisl88/rand_param_envs.git) (already included) must be installed in dev mode on the created env.
For installation, perform the following steps for `meta_rand_envs` (`rand_param_envs` analogously):
```
cd submodules/meta_rand_envs
pip install -e .
```

This installation has been tested only on 64-bit Ubuntu 18.04.

## Training

To train the models in the paper, activate the python env and run these commands:

```train
python runner.py configs/paper/cheetah_multi_task_vel.json
python runner.py configs/paper/cheetah_multi_task_dir.json
python runner.py configs/paper/ant_three.json
python runner.py configs/half_cheetah_eight.json
```

- The experiment results will be stored under `./output/[ENV]/[EXP NAME]`.
- The important parameters for the environment are listed in the respective `json` file. Further parameters can be found in the file `configs/default.py` including small descriptions for each parameter. 
- By default the code will use the GPU - to use CPU instead, set `use_gpu=False` in the corresponding config file.

## Evaluation

We periodically evaluate the algorithm during training. We provide online evaluation via tensorboard located in the `/tensorboard` folder. Use `tensorboard --logdir=./output/[ENV]/[EXP NAME]/tensorboard` for visualizations of learning curves and the current embeddings. For further visual demonstrations after training, add/adjust

```
"path_to_weights": "output/cheetah-multi-task/experiment_name/weights",
"showcase_itr": 2000,
"train_or_showcase": "showcase_all",
```

at the top level of the corresponding training `json` (here: `half_cheetah_eight.json`). Then, activate the python env and run the modified `json` file, e.g. as

```eval
python runner.py configs/paper/cheetah_multi_task.json
```

The generated videos are stored in a new folder under `./output/[ENV]/[EXP NAME]`. Results for task inference results are obtained analogously via setting `"train_or_showcase": "showcase_task_inference"`.

## Pre-trained Models

The weights for a fully-trained `half-cheetah-eight` model are located under `output/comparisons/cheetah-multi-task/ours/2021_05_10_17_35_02_cheetah_multi_task_ts~64_true_gmm/weights`. The used hyper-parameters are stored in the corresponding folder under `variant.json`.

## Results

The results as reported in the paper are visualized under `log/figures`. To reproduce the figures, follow the commands in the file `plotting_configs`.
