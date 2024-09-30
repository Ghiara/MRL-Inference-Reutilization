from sac_envs.walker import WalkerGoal
from sac_envs.hopper import HopperGoal
import numpy as np

pi = 3.141592

config = dict(
    epochs = 30000,
    max_traj_len = 200,
    memory_size = 1e+5,
    batch_size = 256,
    gamma = 0.99,
    alpha = 0.2,
    lr = 3e-4,
    reward_scale = 1,
    change_task_after = 200,
    random_restart_after = 100000000,

    max_goal = [0.2, 10],
    max_jump = [1.5, 3.],
    max_rot = [pi / 6., pi / 2.],
    max_vel = [1.0, 5.0],
    max_rot_vel = [2. * pi, 4. * pi],

    env = 'half_cheetah_pos_add',
    # experiment_name = 'Walker_deeper_really_change_task',
    experiment_name = 'half_cheetah_dt0.03_pos_add_vel_new_arch_256_new2',
    task_dim = 5,

    hidden_layers_actor = [300,300,300],
    hidden_layers_critic = [300,300,300],

    save_after_episodes = 10,

    tasks = dict(
                forward_vel=0, backward_vel=1,
                goal_front=4, 
                goal_back=5
                 ), 



    # pretrained = dict(path = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/half_cheetah_definitive_training',
    #                   epoch = 21300)


)