import stable_baselines3
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn

class ModelLoader:
    def __init__(self, model_file):
        self.model_file = model_file
        self.model = None

    def load_model(self):
        # Define the policy
        policy_kwargs = {
            "log_std_init": -2,
            "ortho_init": False,
            "activation_fn": torch.nn.ReLU,
            "net_arch": [{'pi': [256, 256], 'vf': [256, 256]}]
        }

        # Create the model
        self.model = stable_baselines3.PPO(
            policy=ActorCriticPolicy,
            policy_kwargs=policy_kwargs,
            learning_rate=get_schedule_fn(0.0),
            tensorboard_log=None,
            verbose=1
        )

        # Load the parameters from the file
        self.model.load_state_dict(torch.load(self.model_file))

    def get_model(self):
        if self.model is None:
            self.load_model()
        return self.model

if __name__ == "__main__":
    loader = ModelLoader("/home/ubuntu/juan/Meta-RL/submodules/ppo/HalfCheetah-v3_1/policy.pth")
    model = loader.get_model()