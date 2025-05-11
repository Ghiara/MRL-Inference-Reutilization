# default experiment settings
# all experiments should modify these settings only as needed
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from sac_envs.hopper_multi import HopperMulti
from sac_envs.walker_multi import WalkerMulti

transfer_config = dict(

    experiment_name = 'cheetah_transfer',
    sim_time_steps = 20,
    max_path_len=100,
    batch_size = 20,
    policy_update_steps = 512,
    
    ### Define inference module to be reused
    
    inference_path = '/home/ubuntu/juan/MRL-Inference-Reutilization/output/toy1d-multi-task/2025_03_23_12_18_10_default_true_gmm',

    
    ### Define the low-level controller and agent to reuse the inference mechanism

    complex_agent = dict(
        environment = HalfCheetahMixtureEnv,
        experiments_repo = '/home/ubuntu/juan/MRL-Inference-Reutilization/output/low_level_policy',
        experiment_name = 'new_cheetah_training/half_cheetah_initial_random',
        epoch = 700,
    ),
    # complex_agent = dict(
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'walker_full_06_07',
    #     epoch = 2100,
    #     environment = WalkerMulti,
    # )
    # complex_agent = dict(
    #     environment = HopperMulti,
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'hopper_full_sac0.2_reward1_randomchange',
    #     epoch = 1400,
    # )

)
