from sac_envs.walker import WalkerGoal
from sac_envs.hopper import HopperGoal
import numpy as np

pi = 3.141592

config = dict(
    epochs = 30000,
    max_traj_len = 500,
    memory_size = 1e+6,
    batch_size_memory = 256,
    batch_size = 20,
    policy_update_steps = 2048,
    gamma = 0.99,
    alpha = 0.2,
    lr = 3e-4,
    reward_scale = 1,

    max_goal = [0.2, 10],
    max_jump = [1.5, 3.],
    max_rot = [pi / 6., pi / 2.],
    max_vel = [1.0, 2.5],
    max_rot_vel = [2. * pi, 4. * pi],

    env = 'half_cheetah_multi',
    # experiment_name = 'Walker_deeper_really_change_task',
    experiment_name = 'new_cheetah_training/half_cheetah_initial_random',
    task_dim = 5,

    hidden_layers_actor = [300,300,300,300],
    hidden_layers_critic = [300,300,300,300],

    save_after_episodes = 5,
    plot_every = 10,

    tasks = dict(
                forward_vel=1, backward_vel=1,
                goal_front=0, 
                goal_back=0,
                stand_front= 2, 
                stand_back=2, 
                jump=3, 
                # rotation_back=4,
                # rotation_front=4,
                 ), 

    curriculum = dict(
        max_vel=200,
        change_tasks_after = [100,200,300],
        changes_per_trajectory = [2,3,6],
        max_steps_epochs = [1],
        max_steps = [1000],
        random_initialization = 1,
    ),

    random_restart_after = 1,

    # pretrained = dict(path = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/new_cheetah_training/half_cheetah_21_06',
    #                   file_name = 'epoch_1700')


)