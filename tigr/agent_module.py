import torch
import torch.nn as nn
import numpy as np
from rlkit.torch.core import np_ify
import rlkit.torch.pytorch_util as ptu

# from tigr.scripted_policies import policies

class Agent(nn.Module):
    def __init__(self,
                 encoder,
                 policy,
                 use_sample, 
                 simple_env=False
                 ):
        super(Agent, self).__init__()
        self.encoder = encoder
        self.policy = policy
        self.use_sample = use_sample
        self.simple_env = simple_env

    def get_action(self, encoder_input, state, deterministic=False, z_debug=None, env=None):
        state = ptu.from_numpy(state).view(1, -1)
        if self.use_sample:
            z, _ = self.encoder(encoder_input)
            _, task = torch.max(z, dim=-1)
        else:
            mu, log_var = self.encoder.encode(encoder_input)
            z = torch.cat([mu, log_var], dim=-1)
            _, task = torch.max(mu, dim=-1)
        if z_debug is not None:
            z = z_debug
            _, task = torch.max(z, dim=-1)
        policy_input = torch.cat([state, z], dim=1)
        return self.policy.get_action(policy_input, deterministic=deterministic), np_ify(z.clone().detach())[0, :]

    def get_actions(self, encoder_input, state, deterministic=False, z=None):
        if z is not None:
            z = torch.from_numpy(z)
            _, tasks = torch.max(z, dim=1)
        elif self.use_sample:
            z, _ = self.encoder(encoder_input)
            _, tasks = torch.max(z, dim=1)
        else:
            mu, log_var = self.encoder.encode(encoder_input)
            _, tasks = torch.max(mu, dim=1)
            z = torch.cat([mu, log_var], dim=-1)
        policy_input = torch.cat([state, z], dim=-1)
        actions, info = (self.policy.get_actions(policy_input, deterministic=deterministic), [{}] * state.shape[0])
        # if self.simple_env:
        #     actions_final = np.zeros_like(actions)
        #     x_movement = torch.tensor([4,5], device=tasks.device)
        #     move_in_x_idx = torch.isin(tasks,x_movement).detach().cpu().numpy()
        #     x_vel = torch.tensor([0,1], device=tasks.device)
        #     x_vel_idx = torch.isin(tasks,x_vel).detach().cpu().numpy()
        #     y_movement = torch.tensor([7], device=tasks.device)
        #     move_in_z_idx = torch.isin(tasks,y_movement).detach().cpu().numpy()
        #     rot = torch.tensor([2,3], device=tasks.device)
        #     rot_idx = torch.isin(tasks,rot).detach().cpu().numpy()
        #     flip = torch.tensor([6], device=tasks.device)
        #     flip_idx = torch.isin(tasks,flip).detach().cpu().numpy()
        #     actions_final[move_in_x_idx, 0] = actions[move_in_x_idx, 0]
        #     actions_final[move_in_z_idx, 1] = actions[move_in_z_idx, 1]
        #     actions_final[rot_idx, 2] = actions[rot_idx, 2]
        #     actions_final[x_vel_idx, 3] = actions[x_vel_idx, 3]
        #     actions_final[flip_idx, 4] = actions[flip_idx, 4]

        #     return (actions_final, info), np_ify(z)
        # if self.simple_env:
        #     max_action = np.argmax(actions, axis=-1)
        #     actions[np.arange(np.shape(actions)[0]), max_action] = actions[np.arange(np.shape(actions)[0]), max_action] * 2
        #     actions = actions / 2
        #     return (actions, info), np_ify(z)
        return (actions, info), np_ify(z)


        return (self.policy.get_actions(policy_input, deterministic=deterministic), [{}] * state.shape[0]), np_ify(z)
        
        return (self.policy.get_actions(policy_input, deterministic=deterministic), [{}] * state.shape[0]), np_ify(z)


# class ScriptedPolicyAgent(nn.Module):
#     def __init__(self,
#                  encoder,
#                  policy
#                  ):
#         super(ScriptedPolicyAgent, self).__init__()
#         self.encoder = encoder
#         self.policy = policy
#         self.latent_dim = encoder.latent_dim

#     def get_action(self, encoder_input, state, deterministic=False, z_debug=None, env=None):
#         env_name = env.active_env_name
#         oracle_policy = policies[env_name]()
#         action = oracle_policy.get_action(state)
#         return (action.astype('float32'), {}), np.zeros(self.latent_dim, dtype='float32')

