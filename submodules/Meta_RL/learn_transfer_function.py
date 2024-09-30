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
from datetime import datetime
import pytz
import pdb


class ActionSpaceTransferLearning:
    def __init__(self, simple_env, complex_env, transfer_network, reward_predictor, batch_size, path, max_traj_len, agent, learning_rate=1e-4, pretrained_reward_model=None, reward_predictor_policy='transfer', only_reward_epochs=99):
        self.simple_env = simple_env
        self.complex_env = complex_env
        self.complex_env.render_mode = None
        self.transfer_network = transfer_network
        self.reward_predictor = reward_predictor
        if torch.cuda.is_available():
            self.transfer_network = self.transfer_network.cuda()
            self.reward_predictor = self.reward_predictor.cuda()
        self.optimizer_reward = optim.Adam(self.reward_predictor.parameters(), lr=learning_rate)
        self.optimizer_policy = optim.Adam(self.transfer_network.parameters(), lr=learning_rate)
        self.reward_loss_history, self.policy_loss_history = [], []
        self.batch_size = batch_size
        self.path = path
        self.max_policy_traj_len = max_traj_len
        self.max_rew_traj_len = max_traj_len
        self.max_traj_len = max_traj_len
        self.agent = agent
        self.date = datetime.now().astimezone(pytz.timezone('Europe/Berlin'))
        self.pretrained_reward_model = pretrained_reward_model
        if reward_predictor_policy not in ['random', 'transfer']:
            print('Invalid argument for reward_predictor_policy. Using the transfer function to create actions')
            self.reward_predictor_policy = 'transfer'
        else:
            self.reward_predictor_policy = reward_predictor_policy
        self.only_reward_epochs = only_reward_epochs

    def train(self, epochs):
        
        loss_function = nn.MSELoss()

        for episode in range(epochs):
            print("\033[1m" + f"Starting Epoch {episode}" + "\033[0m")
            for param in self.transfer_network.parameters():
                param.requires_grad = False
            for param in self.reward_predictor.parameters():
                param.requires_grad = True
            policy_loss, reward_loss = [], []

            #TODO: change to logger
            print("Training reward estimator")

            # Train reward predictor
            for _ in tqdm(range(self.batch_size)):

                o_complex, complex_rewards, done_complex, trunc_complex, env_info_complex = [], [], [], [], []
                real_rewards, predicted_rewards, reward_traj_losses = [], [], []

                #TODO: start also with random positions, but they need to be not too random, otherwise the robot will end up in weird positions which are of no value
                o_complex.append(self.complex_env.reset()[0])
                self.simple_env.reset()
                reward_est_traj_len = random.randint(1,self.max_rew_traj_len)
                for i in range(reward_est_traj_len):
                    self.complex_env.update_task(self.simple_env.task)
                    # simple_action = torch.from_numpy(np.array([self.agent.get_action(self.simple_env.observation)[0][0] for _ in range(traj_len)])).float()   # obs only passed to get the shape of obs -> Assumes action is same shape as observation?
                    simple_action = torch.from_numpy(np.array(self.agent.get_action(self.simple_env.observation)[0][0])).float().cuda()

                    if self.reward_predictor_policy == 'transfer' and (episode>self.only_reward_epochs or (self.pretrained_reward_model is not None)):
                        complex_action = self.transfer_network.forward(torch.tensor(o_complex[i]).cuda(), simple_action)
                        step_result = self.complex_env.step(complex_action.detach().cpu().numpy())  # Assuming simple_actions are applicable
                    else:
                        complex_action = self.complex_env.action_space.sample()
                        step_result = self.complex_env.step(complex_action)  # Assuming simple_actions are applicable
                        complex_action = torch.from_numpy(complex_action).cuda()

    
                    o_complex.append(step_result[0])
                    complex_rewards.append(step_result[1])
                    done_complex.append(step_result[2])
                    trunc_complex.append(step_result[3])
                    env_info_complex.append(step_result[4])
                    real_rewards.append(step_result[1])         #/self.simple_env.max_pos)
                    predicted_rewards.append(self.reward_predictor.forward(torch.tensor(o_complex[i]).cuda(),
                                                                        complex_action,        # Changed for sample action
                                                                        torch.from_numpy(self.complex_env.task['goal']).cuda()))    #/torch.tensor(self.simple_env.max_pos, dtype=torch.float).cuda())
                    
                stacked_predicted_rewards = torch.stack(predicted_rewards).float()
                reward_traj_losses.append(loss_function(stacked_predicted_rewards.squeeze(), torch.tensor(real_rewards, dtype=torch.float32).cuda().squeeze()))
            reward_mean_loss = torch.mean(torch.stack(reward_traj_losses), dtype=torch.float32)

            # initial_weights = {name: param.clone() for name, param in self.reward_predictor.named_parameters()}
            self.optimizer_reward.zero_grad()
            reward_mean_loss.backward()
            self.optimizer_reward.step()
            self.reward_loss_history.append(reward_mean_loss.item())

            #TODO: Make the creation of the repo more robust
            if episode % 10 == 0 and episode!=0:
                path = self.path + self.date.strftime('%Y-%m-%d_%H-%M-%S') + '/reward_model/'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                save_path = path + f'epoch_{episode}.pth'
                if episode % 50 == 0:
                    torch.save(self.reward_predictor.cpu(), save_path)
                self.reward_predictor.cuda() 
                new_repo = f'{self.date.strftime("%Y-%m-%d_%H-%M-%S")}/'
                save_plot(self.reward_loss_history, name='reward_loss', new_repo=new_repo)

            # Train tansfer_network
            if episode > self.only_reward_epochs or self.pretrained_reward_model is not None:
                for param in self.transfer_network.parameters():
                    param.requires_grad = True
                for param in self.reward_predictor.parameters():
                    param.requires_grad = False
                print("Training hierarchical policy module")
                self.max_policy_traj_len = np.clip(int(episode/10),1,self.max_traj_len)
                for _ in tqdm(range(self.batch_size)):
                    o_simple, simple_rewards, done_simple, trunc_simple, env_info_simple = [], [], [], [], []
                    o_complex, complex_rewards, done_complex, trunc_complex, env_info_complex = [], [], [], [], []

                    self.simple_env.reset()
                    self.simple_env.sample_task()
                    o_complex = self.complex_env.reset()[0]
                    simple_action_add = 0
                    simple_action = torch.from_numpy(np.array(self.agent.get_action(self.simple_env.observation)[0][0])).float().cuda()
                    for i in range(10):
                        step_result_simple = self.simple_env.step(simple_action)
                        simple_reward = step_result_simple[1]               #/self.simple_env.max_pos
                    
                    # policy_traj_len = random.randint(1,self.max_policy_traj_len)
                    policy_traj_len = 25
                    
                    # simple_actions = torch.from_numpy(np.array([self.agent.get_action(self.simple_env.observation)[0][0] for _ in range(policy_traj_len)])).float()   # obs only passed to get the shape of obs -> Assumes action is same shape as observation?
                    # if torch.cuda.is_available():
                    #     simple_actions = simple_actions.cuda()
                    for i in range(policy_traj_len):
                        # step_result = self.simple_env.step(simple_action)
                        # simple_reward = step_result[1]/self.simple_env.max_pos
                        
                        # o_simple.append(step_result[0])
                        # simple_rewards.append(step_result[1])
                        # done_simple.append(step_result[2])
                        # trunc_simple.append(step_result[3])
                        # env_info_simple.append(step_result[4])
                        simple_action = torch.from_numpy(np.array(step_result_simple[0][0])).float().cuda() - o_complex[0]
                        complex_action = self.transfer_network.forward(torch.tensor(o_complex).cuda(), simple_action).float()
        
                        #TODO: pass through reward estimator and compare

                        self.complex_env.update_task(self.simple_env.task)
                        predicted_reward = self.reward_predictor.forward(torch.tensor(o_complex).cuda(),
                                                    complex_action, 
                                                    torch.from_numpy(self.complex_env.task['goal']).cuda())         #/torch.tensor(self.simple_env.max_pos, dtype=torch.float).cuda()
                        step_result = self.complex_env.step(complex_action.detach().cpu().numpy())  # Assuming simple_actions are applicable
                        o_complex = step_result[0]

                    policy_traj_loss = loss_function(torch.tensor(simple_reward, dtype=torch.float32).unsqueeze(0).cuda(), predicted_reward).float()
                    policy_loss.append(policy_traj_loss)

                # initial_weights = {name: param.clone() for name, param in self.transfer_network.named_parameters()}
                policy_mean_loss = torch.mean(torch.stack(policy_loss).float(), dtype=torch.float32)
                self.optimizer_policy.zero_grad()
                policy_mean_loss.backward()
                self.optimizer_policy.step()
                self.policy_loss_history.append(policy_mean_loss.item())

                # for name, param in self.transfer_network.named_parameters():
                #     print('i entered')
                #     print(param.requires_grad)
                #     if torch.equal(initial_weights[name], param):
                #         print(f"Weights stayed same for layer: {name}")


                #TODO: Make the creation of the repo more robust
                if episode % 10 == 0 and episode!=0:
                    path = self.path + self.date.strftime('%Y-%m-%d_%H-%M-%S') + '/policy_model/'
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    save_path = path + f'epoch_{episode}.pth'
                    if episode % 50 == 0:
                        torch.save(self.transfer_network.cpu(), save_path)
                    self.transfer_network.cuda() 
                    new_repo = f'{self.date.strftime("%Y-%m-%d_%H-%M-%S")}/'
                    save_plot(self.policy_loss_history, name='policy_loss', new_repo=new_repo)

def save_plot(loss_history, name:str, new_repo, path='/home/ubuntu/juan/Meta-RL/evaluation/transfer_function/traj_for_policy/'):
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

    config = OrderedDict(
        
        save_model_path = '/home/ubuntu/juan/Meta-RL/data/transfer_function/traj_for_policy/',

        obs_dim_complex = 20,
        act_simple_dim = 1,
        act_complex_dim = 6,
        task_dim = 1,
        
        hidden_sizes_transfer = [256,256,256],
        hidden_sizes_critic = [256,256,256],

        train_epochs = 5_000,
        batch_size = 25,
        max_traj_len = 100,
        only_reward_epochs = 150,

        # pretrained_reward_predictor = '/home/ubuntu/juan/Meta-RL/data/transfer_function/traj_for_policy/2023-12-17_14-37-29/reward_model/epoch_100.pth',
        pretrained_reward_predictor = None,
        pretrained_transfer_function = None,

        # can either be random or use the transfer_network to create actions
        reward_predictor_policy = 'transfer',
    )

    #TODO: Rewrite this such that it uses a base funcion and argumetns are inputs, not abs_dim,...
    transfer_network = TransferFunction(obs_dim=config['obs_dim_complex'], 
                                        act_simple_dim=config['act_simple_dim'], 
                                        hidden_sizes=config['hidden_sizes_transfer'], 
                                        act_complex_dim=config['act_complex_dim'],
                                        )

    reward_predictor = RewardPredictor(act_complex_dim=config['act_complex_dim'],
                                      obs_dim=config['obs_dim_complex'],
                                      hidden_sizes=config['hidden_sizes_critic'],
                                      task_dim=config['task_dim'],
                                      reward_dim=1,
                                      pretrained=config['pretrained_reward_predictor']
                                      )


    env_args = []
    env_kwargs = {
        "n_train_tasks": 100,
        "n_eval_tasks": 25,
        "change_steps": 500,
        "task_generation_mode": 'random',
        "min_pos": -50,
        "max_pos": 50,
        "max_action": 1.0,
    }
    simple_env = Toy1D(*env_args, **env_kwargs)
    agent = MultiRandomMemoryPolicy(action_dim=1, action_update_interval=25, std_mean_range=(0.05,0.1)) # TODO: try differnet values

    complex_env = HalfCheetahEnvExternalTask(1,1) ## 1,1 only such that super().__init__ works

    # After training
    alg = ActionSpaceTransferLearning(simple_env, 
                                      complex_env, 
                                      transfer_network, 
                                      reward_predictor, 
                                      batch_size=config['batch_size'], 
                                      path=config['save_model_path'], 
                                      agent=agent, 
                                      max_traj_len=config['max_traj_len'], 
                                      pretrained_reward_model=config['pretrained_reward_predictor'], 
                                      reward_predictor_policy=config['reward_predictor_policy'],
                                      only_reward_epochs=config['only_reward_epochs'],
                                      )
    alg.train(config['train_epochs'])

    torch.save(alg.transfer_network.state_dict(), config['save_model_path']+'policy_model/' + alg.date.strftime('%Y-%m-%d_%H-%M-%S') + '/''final_model.pth')
    torch.save(alg.reward_predictor.state_dict(), config['save_model_path']+'reward_model/' + alg.date.strftime('%Y-%m-%d_%H-%M-%S') + '/''final_model.pth')


