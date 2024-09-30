from collections import OrderedDict
from meta_envs.mujoco.hopper import HopperGoal

config = OrderedDict(
        
        save_model_path = '/home/ubuntu/juan/Meta-RL/data/transfer_function/hopper/',

        environment = HopperGoal(),

        obs_dim_complex = 12,
        act_simple_dim = 1,
        act_complex_dim = 3,
        task_dim = 1,
        
        hidden_sizes_transfer = [300,300,300],
        hidden_sizes_critic = [300,300,300],

        train_epochs = 20_000,
        batch_size = 256,
        max_traj_len = 1000,
        only_reward_epochs = 0,
        max_path_len = 500,

        # pretrained_vf1= '/home/ubuntu/juan/Meta-RL/data/transfer_function/hopper/2024-01-28_09-47-24/vf1_model/best_model(56).pth',
        pretrained_vf1 = None,
        # pretrained_vf2= '/home/ubuntu/juan/Meta-RL/data/transfer_function/hopper/2024-01-28_09-47-24/vf1_model/best_model(56).pth',
        pretrained_vf2 = None,
        # pretrained_transfer_function = '/home/ubuntu/juan/Meta-RL/data/transfer_function/hopper/2024-01-31_12-08-02/policy_model/epoch_220.pth',
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