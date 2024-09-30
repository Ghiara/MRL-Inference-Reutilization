from sac_envs.walker import WalkerGoal
from sac_envs.hopper import HopperGoal
import numpy as np

pi = 3.141592

config = dict(
    epochs = 30000,
    max_traj_len = 300,
    memory_size = 1e+6,
    batch_size = 20,
    batch_size_memory = 128,
    policy_update_steps = 1024,
    gamma = 0.99,
    alpha = 0.2,
    lr = 3e-4,
    reward_scale = 5,

    max_goal = [2, 15],
    max_jump = [1.5, 3.],
    max_rot = [pi / 6., pi / 2.],
    max_vel = [1.0, 5.0],
    max_rot_vel = [2. * pi, 4. * pi],

    env = 'walker_multi',
    # experiment_name = 'Walker_deeper_really_change_task',
    experiment_name = 'walker_full_06_07',
    # task_dim = 2,

    tasks = dict(
                goal_front=0, goal_back=0, 
                forward_vel=1, backward_vel=1
                 ),

    hidden_layers_actor = [300,300,300,300],
    hidden_layers_critic = [300,300,300,300],

    save_after_episodes = 20,
    plot_every = 100,

    # curriculum = dict(
    #     max_vel=500,
    #     change_tasks_after = [200,300,400],
    #     changes_per_trajectory = [2,3,4],
    # )

    curriculum = dict(
        max_vel=500,
        change_tasks_after = [200,300,400,600,1000],
        changes_per_trajectory = [0,2,4,6,8],
        max_steps_epochs = [200,400,600],
        max_steps = [300,600,1000],
        random_initialization = 400,
    ),

    # pretrained=dict(
    #     path = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/walker_back_to_roots_new_arc',
    #     epoch = 10000,
    # )

)