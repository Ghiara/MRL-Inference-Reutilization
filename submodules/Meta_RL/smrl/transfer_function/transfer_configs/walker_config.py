from collections import OrderedDict
from meta_envs.mujoco.walker2d import WalkerGoal
import os

config = OrderedDict(
        
        save_model_path = f'{os.getcwd()}/data/transfer_function/walker/',

        environment = WalkerGoal(),

        obs_dim_complex = 18,
        act_simple_dim = 1,
        act_complex_dim = 6,
        task_dim = 1,
        
        hidden_sizes_transfer = [300,300,300],
        hidden_sizes_critic = [300,300,300],

        train_epochs = 1_000,
        batch_size = 25,
        max_traj_len = 50,
        only_reward_epochs = 0,
        max_path_len = 1000,

        # pretrained_vf1 = '/home/ubuntu/juan/Meta-RL/data/transfer_function/walker/2024-01-28_22-05-44/vf1_model/epoch_470.pth',
        # pretrained_vf2 = '/home/ubuntu/juan/Meta-RL/data/transfer_function/walker/2024-01-28_22-05-44/vf2_model/epoch_470.pth',
        pretrained_vf1 = None,
        pretrained_vf2 = None,
        pretrained_transfer_function = '/home/ubuntu/juan/Meta-RL/data/transfer_function/walker/2024-01-29_11-07-59/policy_model/best_model(637).pth',
        # pretrained_transfer_function = None,

        # can either be random or use the transfer_network to create actions
        reward_predictor_policy = 'transfer',
        extended_reward = False,

        sample_until_terminal=True,

        random_change_task = dict(
            prob = 0.7,
            factor = 2,
        )
        # random_change_task = None
    )