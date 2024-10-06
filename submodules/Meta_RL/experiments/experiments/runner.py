from typing import Dict, Any
import run_toy_training

def run_experiment(
    config: Dict[str, Any], 
    gpu: str = "",
    multithreading: bool = True,
    save_env: bool = False,
    temp_dir: str = "/home/ubuntu/juan/Meta-RL/data/tmp/ray",
    log_epoch_gap: int = 50,
    log_dir: str = '/home/ubuntu/juan/Meta-RL/data',
):
    run_toy_training.main(
        environment_factory=config['environment_factory'], 
        config=config, 
        log_dir=log_dir,
        gpu=gpu, 
        multithreading=multithreading, 
        save_env=save_env,
        verbose = True,
        temp_dir=temp_dir,
        log_epoch_gap=log_epoch_gap,
    )