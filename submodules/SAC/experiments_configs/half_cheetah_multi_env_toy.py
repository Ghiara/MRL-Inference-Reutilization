from sac_envs.walker import WalkerGoal
from sac_envs.hopper import HopperGoal
import numpy as np

pi = 3.141592

config = dict(
    epochs = 30000,
    max_traj_len = 100,
    memory_size = 1e+6,
    batch_size = 256,
    gamma = 0.99,
    alpha = 0.5,
    lr = 3e-5,
    reward_scale = 1,

    max_goal = [2, 15],
    max_jump = [1.5, 3.],
    max_rot = [pi / 6., pi / 2.],
    max_vel = [1.0, 5.0],
    max_rot_vel = [2. * pi, 4. * pi],

    env = 'half_cheetah_follow_toy',
    # experiment_name = 'Walker_deeper_really_change_task',
    experiment_name = 'half_cheetah_follow_toy',
    task_dim = 5,

    hidden_layers_actor = [300,300,300],
    hidden_layers_critic = [300,300,300],

    save_after_episodes = 10,


)