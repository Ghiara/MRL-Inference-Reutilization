from collections import OrderedDict
from meta_envs.mujoco.swimmer import SwimmerGoal

env_kwargs = {
    "render_mode": 'rgb_array',
}
env_args = []

config = OrderedDict(
        
        save_model_path = '/home/ubuntu/juan/Meta-RL/data/transfer_function/swimmer/',

        environment = SwimmerGoal(*env_args, **env_kwargs),

        obs_dim_complex = 10,
        act_simple_dim = 1,
        act_complex_dim = 2,
        task_dim = 1,
        
        hidden_sizes_transfer = [300,300,300],
        hidden_sizes_critic = [300,300,300],

        train_epochs = 1_000,
        batch_size = 256,
        max_traj_len = 20,
        only_reward_epochs = 0,
        max_path_len = 50,

        # pretrained_reward_predictor = '/home/ubuntu/juan/Meta-RL/data/transfer_function/one-sided/2024-01-16_16-25-43/vf1_model/epoch_1000.pth',
        pretrained_reward_predictor = None,
        # pretrained_transfer_function = '/home/ubuntu/juan/Meta-RL/data/transfer_function/extended_reward/random_change_task/prob_0.7/2024-01-19_11-30-20/policy_model/epoch_1150.pth',
        pretrained_transfer_function = None,

        # can either be random or use the transfer_network to create actions
        reward_predictor_policy = 'transfer',
        extended_reward = False,

        sample_until_terminal=False,

        random_change_task = dict(
            prob = 0.7,
            factor = 2,
        )
        # random_change_task = None
    )