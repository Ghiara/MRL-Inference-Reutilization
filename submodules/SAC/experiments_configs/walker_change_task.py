from sac_envs.walker import WalkerGoal



config = dict(
    epochs = 20000,
    max_traj_len = 1000,
    memory_size = 1e+6,
    batch_size = 256,
    gamma = 0.99,
    alpha = 1,
    lr = 3e-4,
    reward_scale = 5,

    env = 'walker',
    # experiment_name = 'Walker_deeper_really_change_task',
    experiment_name = 'delete2',
    task_dim = 1,

    hidden_layers_actor = [256,256],
    hidden_layers_critic = [256,256,256,256],

    save_after_episodes = 50,


)