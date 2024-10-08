import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from collections import OrderedDict

from configs.transfer_functions_config import transfer_config
from smrl.trainers.transfer_function import TransferFunction, RewardPredictor
from meta_envs.toy_goal import Toy1D
from meta_envs.mujoco.cheetah import HalfCheetahEnvExternalTask
from smrl.policies.exploration import RandomPolicy, RandomMemoryPolicy, MultiRandomMemoryPolicy
from smrl.transfer_function.PathGenerator import RolloutCoordinator
from smrl.transfer_function.stacked_replay_buffer import StackedReplayBuffer
from datetime import datetime
import pytz
import rlkit.torch.pytorch_util as ptu
import pdb
torch.autograd.set_detect_anomaly(True)

class ActionSpaceTransferLearning:
    def __init__(self, simple_env, complex_env, transfer_network, vf1, vf2, batch_size, path, max_traj_len, agent, stacked_replay_buffer, policy_updates=100, num_tasks=25, range_tasks=[-5,5], learning_rate=3e-4, alpha=0.2, gamma=0.99, target_update_period=1, pretrained_reward_model=None, reward_predictor_policy='transfer', only_reward_epochs=99):
        self.simple_env = simple_env
        self.complex_env = complex_env
        self.complex_env.render_mode = None
        self.transfer_network = transfer_network
        self.vf1 = vf1
        self.target_vf1 = vf1
        self.target_vf2 = vf2
        self.vf2 = vf2
        self.soft_target_tau = 5e-3
        self.alpha = alpha
        self.gamma = gamma
        self.num_tasks = num_tasks
        self.range_tasks = range_tasks
        self.policy_updates = policy_updates
        self.vf_criterion = nn.MSELoss(reduction='none')
        self.policy_criterion = nn.MSELoss(reduction='none')
        if torch.cuda.is_available():
            self.transfer_network = self.transfer_network.cuda()
            self.vf1 = self.vf1.cuda()
            self.vf2 = self.vf2.cuda()
        self.optimizer_vf1 = optim.Adam(self.vf1.parameters(), lr=learning_rate, weight_decay=0.01)
        self.optimizer_vf2 = optim.Adam(self.vf2.parameters(), lr=learning_rate, weight_decay=0.01)
        self.optimizer_policy = optim.Adam(self.transfer_network.parameters(), lr=learning_rate)
        self.vf1_loss_history, self.vf2_loss_history, self.policy_loss_history, self.reward_history = [], [], [], []
        self.batch_size = batch_size
        self.path = path
        self.max_policy_traj_len = max_traj_len
        self.max_rew_traj_len = max_traj_len
        self.max_traj_len = max_traj_len
        self.agent = agent
        self.date = datetime.now().astimezone(pytz.timezone('Europe/Berlin'))
        self.pretrained_reward_model = pretrained_reward_model
        self.replay_buffer = stacked_replay_buffer
        self._n_train_steps_total = 0
        self.target_update_period = target_update_period
        self.best_reward = dict()
        if reward_predictor_policy not in ['random', 'transfer']:
            print('Invalid argument for reward_predictor_policy. Using the transfer function to create actions')
            self.vf_policy = 'transfer'
        else:
            self.vf_policy = reward_predictor_policy
        self.only_reward_epochs = only_reward_epochs

    def train(self, epochs):

        for episode in range(epochs):

            print("\033[1m" + f"Starting Epoch {episode}" + "\033[0m")
            random_tasks = np.random.uniform(self.range_tasks[0], self.range_tasks[1], self.num_tasks)
            self.replay_buffer.delete_buffer()
            self.replay_buffer.fill_buffer(random_tasks)
            rewards = self.replay_buffer.results['rewards']
            print('Learning the transfer function')
            for i in tqdm(range(self.policy_updates)):

                sac_data = self.replay_buffer.sample_batch(self.batch_size, self.max_traj_len)
                rewards_batch = ptu.from_numpy(sac_data['rewards']).cuda()
                terminals = ptu.from_numpy(sac_data['terminals']).cuda()
                obs = ptu.from_numpy(sac_data['observations']).cuda()
                actions = ptu.from_numpy(sac_data['actions']).cuda()
                tasks = ptu.from_numpy(sac_data['tasks']).cuda()
                next_obs = ptu.from_numpy(sac_data['next_observations']).cuda()

                new_actions, log_probs = self.transfer_network.get_action(obs, tasks, return_dist=True)
                q_new_actions = torch.min(
                    self.vf1.forward(obs,
                                     new_actions,
                                     tasks), 
                    self.vf2.forward(obs, 
                                     new_actions,
                                     tasks)
                )
                policy_loss = (self.alpha*log_probs[:, None] - q_new_actions)

                q1_pred = self.vf1.forward(obs, 
                                           actions,
                                           tasks)
                q2_pred = self.vf2.forward(obs,
                                           actions,
                                           tasks)
                new_next_actions,new_log_probs = self.transfer_network.get_action(next_obs, tasks, return_dist=True)
                target_q_values = torch.min(
                    self.target_vf1.forward(next_obs, new_next_actions, tasks),
                    self.target_vf2.forward(next_obs, new_next_actions, tasks)
                ) - self.alpha*new_log_probs[:, None]
                q_target = rewards_batch[:, None] + (1. - terminals[:, None]) * self.gamma * target_q_values
                qf1_loss = self.vf_criterion(q1_pred, q_target.detach())
                qf2_loss = self.vf_criterion(q2_pred, q_target.detach())

                policy_loss = policy_loss.mean()
                # policy_loss = torch.clip(policy_loss, -10_000, 10_000)
                qf1_loss = qf1_loss.mean()
                # qf1_loss = torch.clip(qf1_loss, -10_000, 10_000)
                qf2_loss = qf2_loss.mean()
                # qf2_loss = torch.clip(qf2_loss, -10_000, 10_000)
                self.vf1_loss_history.append(qf1_loss.detach().cpu().numpy())
                self.policy_loss_history.append(policy_loss.detach().cpu().numpy())

                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()

                self.optimizer_vf1.zero_grad()
                qf1_loss.backward()
                self.optimizer_vf1.step()

                self.optimizer_vf2.zero_grad()
                qf2_loss.backward()
                self.optimizer_vf2.step()

                if self._n_train_steps_total % self.target_update_period == 0:
                    ptu.soft_update_from_to(
                        self.vf1, self.target_vf1, self.soft_target_tau
                    )
                    ptu.soft_update_from_to(
                        self.vf2, self.target_vf2, self.soft_target_tau
                    )
                self._n_train_steps_total+=1
            self.reward_history.append(np.mean(rewards))

            #TODO: Make the creation of the repo more robust
            if episode % 10 == 0 and episode!=0:
                path = self.path + self.date.strftime('%Y-%m-%d_%H-%M-%S') + '/policy_model/'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                save_path = path + f'epoch_{episode}.pth'
                if episode % 10 == 0:
                    torch.save(self.transfer_network.cpu(), save_path)
                self.transfer_network.cuda() 
                new_repo = f'{self.date.strftime("%Y-%m-%d_%H-%M-%S")}/'
                save_plot(self.policy_loss_history, name='policy_loss', new_repo=new_repo)

                path = self.path + self.date.strftime('%Y-%m-%d_%H-%M-%S') + '/vf1_model/'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                save_path = path + f'epoch_{episode}.pth'
                if episode % 10 == 0:
                    torch.save(self.vf1.cpu(), save_path)
                path = self.path + self.date.strftime('%Y-%m-%d_%H-%M-%S') + '/vf2_model/'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                save_path = path + f'epoch_{episode}.pth'
                if episode % 10 == 0:
                    torch.save(self.vf2.cpu(), save_path)
                self.vf1.cuda() 
                self.vf2.cuda()
                new_repo = f'{self.date.strftime("%Y-%m-%d_%H-%M-%S")}/'
                save_plot(self.vf1_loss_history, name='vf_loss', new_repo=new_repo)
                save_plot(self.reward_history, name='reward_history', new_repo=new_repo)

            # TODO: Rewrite
            if episode > 10:
                if episode == 11:
                    self.best_reward['reward'] = max(self.reward_history[:])
                    self.best_reward['episode'] = episode
                    save_path  = self.path + self.date.strftime('%Y-%m-%d_%H-%M-%S') + '/policy_model/' + f'best_model({episode}).pth'
                    torch.save(self.transfer_network.cpu(), save_path)
                    self.transfer_network.cuda()
                    save_path  = self.path + self.date.strftime('%Y-%m-%d_%H-%M-%S') + '/vf1_model/' + f'best_model({episode}).pth'
                    torch.save(self.vf1.cpu(), save_path)
                    self.vf1.cuda()
                if np.mean(self.reward_history[-5:-1])>self.best_reward['reward']:
                    remove_path  = self.path + self.date.strftime('%Y-%m-%d_%H-%M-%S') + '/policy_model/' + f'best_model({self.best_reward["episode"]}).pth'
                    os.remove(remove_path)
                    save_path  = self.path + self.date.strftime('%Y-%m-%d_%H-%M-%S') + '/policy_model/' + f'best_model({episode}).pth'
                    torch.save(self.transfer_network.cpu(), save_path)
                    self.transfer_network.cuda()
                    remove_path  = self.path + self.date.strftime('%Y-%m-%d_%H-%M-%S') + '/vf1_model/' + f'best_model({self.best_reward["episode"]}).pth'
                    os.remove(remove_path)
                    save_path  = self.path + self.date.strftime('%Y-%m-%d_%H-%M-%S') + '/vf1_model/' + f'best_model({episode}).pth'
                    torch.save(self.vf1.cpu(), save_path)
                    self.vf1.cuda()
                    self.best_reward['reward'] = self.reward_history[-1]
                    self.best_reward['episode'] = episode
                
            
# TODO: save both vf losses (maybe with arg)
def save_plot(loss_history, name:str, new_repo, path=f'{os.getcwd()}/evaluation/transfer_function/one-sided/'):
    path = path + new_repo
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure()
    # Plotting the loss
    plt.plot(loss_history)
    plt.title('Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(path+name+'.png')

if __name__=="__main__":

    from transfer_function.transfer_configs.hopper_config import config

    #TODO: Rewrite this such that it uses a base funcion and argumetns are inputs, not abs_dim,...
    transfer_network = TransferFunction(obs_dim=config['obs_dim_complex'], 
                                        act_simple_dim=config['act_simple_dim'], 
                                        hidden_sizes=config['hidden_sizes_transfer'], 
                                        act_complex_dim=config['act_complex_dim'],
                                        pretrained=config['pretrained_transfer_function']
                                        )

    vf1 = RewardPredictor(act_complex_dim=config['act_complex_dim'],
                                      obs_dim=config['obs_dim_complex'],
                                      hidden_sizes=config['hidden_sizes_critic'],
                                      task_dim=config['task_dim'],
                                      reward_dim=1,
                                      pretrained=config['pretrained_vf1']
                                      )
    vf2 = RewardPredictor(act_complex_dim=config['act_complex_dim'],
                                      obs_dim=config['obs_dim_complex'],
                                      hidden_sizes=config['hidden_sizes_critic'],
                                      task_dim=config['task_dim'],
                                      reward_dim=1,
                                      pretrained=config['pretrained_vf2']
                                      )


    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "min_pos": 0,
        "max_pos": 1,
        "max_action": 0.1,
    }
    simple_env = Toy1D(*env_args, **env_kwargs)
    agent = MultiRandomMemoryPolicy(action_dim=1, action_update_interval=25, std_mean_range=(0.05,0.1)) # TODO: try differnet values

    complex_env = config['environment'] ## 1,1 only such that super().__init__ works
    replay_buffer = StackedReplayBuffer(complex_env, 
                                        transfer_network, 
                                        observation_dim=config['obs_dim_complex'], 
                                        action_dim=config['act_complex_dim'], 
                                        random_change_task=config['random_change_task'],
                                        sample_until_terminal=config['sample_until_terminal'],
                                        max_path_length=config['max_path_len'])
    # path_generator = RolloutCoordinator(complex_env)
    # After training
    alg = ActionSpaceTransferLearning(simple_env, 
                                      complex_env, 
                                      transfer_network, 
                                      vf1,
                                      vf2, 
                                      stacked_replay_buffer=replay_buffer,
                                      batch_size=config['batch_size'], 
                                      path=config['save_model_path'], 
                                      agent=agent, 
                                      max_traj_len=config['max_traj_len'], 
                                      reward_predictor_policy=config['reward_predictor_policy'],
                                      only_reward_epochs=config['only_reward_epochs'],
                                      )
    
    alg.train(config['train_epochs'])

    torch.save(alg.transfer_network.state_dict(), config['save_model_path']+'policy_model/' + alg.date.strftime('%Y-%m-%d_%H-%M-%S') + '/''final_model.pth')
    torch.save(alg.vf1.state_dict(), config['save_model_path']+'reward_model/' + alg.date.strftime('%Y-%m-%d_%H-%M-%S') + '/''final_model.pth')


