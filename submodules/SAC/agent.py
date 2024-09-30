import numpy as np
from model import PolicyNetwork, QvalueNetwork, ValueNetwork
import torch
from replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
import matplotlib.pyplot as plt

import os
from datetime import datetime
import pytz

class SAC:
    def __init__(self, n_states, n_actions, task_dim, hidden_layers_actor, hidden_layers_critic, memory_size, batch_size, gamma, alpha, lr, action_bounds,
                 reward_scale, path='/home/ubuntu/juan/Meta-RL/data/transfer_function/hopper/', out_actions=None, pretrained=None, device=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.task_dim = task_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.action_bounds = action_bounds
        self.reward_scale = reward_scale
        self.memory = Memory(memory_size=self.memory_size)

        self.out_actions = out_actions
        if device:
            if device=='cuda':
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device='cpu'
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if pretrained:
            policy_path = f"{pretrained['path']}/models/policy_model/{pretrained['file_name']}.pth"
            self.policy_network = PolicyNetwork(n_states=self.n_states, n_actions=self.n_actions,
                                            action_bounds=self.action_bounds, task_dim=task_dim, 
                                            hidden_layers=hidden_layers_actor, out_actions=out_actions, pretrained=policy_path).to(self.device)
            if out_actions:
                self.n_actions = out_actions
            q1_path = f"{pretrained['path']}/models/vf1_model/{pretrained['file_name']}.pth"
            self.q_value_network1 = QvalueNetwork(n_states=self.n_states, n_actions=self.n_actions, 
                                                task_dim=task_dim,hidden_layers=hidden_layers_critic, pretrained=q1_path).to(self.device)
            q2_path = f"{pretrained['path']}/models/vf2_model/{pretrained['file_name']}.pth"
            self.q_value_network2 = QvalueNetwork(n_states=self.n_states, n_actions=self.n_actions, 
                                                task_dim=task_dim, hidden_layers=hidden_layers_critic, pretrained=q2_path).to(self.device)
            value_path = f"{pretrained['path']}/models/value_model/{pretrained['file_name']}.pth"
            self.value_network = ValueNetwork(n_states=self.n_states, task_dim=task_dim, hidden_layers=hidden_layers_critic, pretrained = value_path).to(self.device)
            self.value_target_network = ValueNetwork(n_states=self.n_states, task_dim=task_dim, hidden_layers=hidden_layers_critic, pretrained = value_path).to(self.device)
            # self.value_target_network.load_state_dict(self.value_network.state_dict())
            self.value_target_network.eval()

        else:
            self.policy_network = PolicyNetwork(n_states=self.n_states, n_actions=self.n_actions,
                                                action_bounds=self.action_bounds, task_dim=task_dim, 
                                                hidden_layers=hidden_layers_actor, out_actions=out_actions).to(self.device)
            if out_actions:
                self.n_actions = out_actions
            self.q_value_network1 = QvalueNetwork(n_states=self.n_states, n_actions=self.n_actions, 
                                                task_dim=task_dim,hidden_layers=hidden_layers_critic).to(self.device)
            self.q_value_network2 = QvalueNetwork(n_states=self.n_states, n_actions=self.n_actions, 
                                                task_dim=task_dim, hidden_layers=hidden_layers_critic).to(self.device)
            self.value_network = ValueNetwork(n_states=self.n_states, task_dim=task_dim, hidden_layers=hidden_layers_critic).to(self.device)
            self.value_target_network = ValueNetwork(n_states=self.n_states, task_dim=task_dim, hidden_layers=hidden_layers_critic).to(self.device)
            self.value_target_network.load_state_dict(self.value_network.state_dict())
            self.value_target_network.eval()

        self.value_loss = torch.nn.MSELoss()
        self.q_value_loss = torch.nn.MSELoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.lr)
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.lr)
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.lr)
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.lr)

        self.policy_loss_history = []
        self.vf1_loss_history = []
        self.reward_history = []
        self.best_reward = dict()
        self.date = datetime.now().astimezone(pytz.timezone('Europe/Berlin'))
        self.path = path
        self.repos_created = False

    def store(self, state, reward, done, action, next_state, task):
        state = from_numpy(state).float().to("cpu")
        reward = torch.Tensor([reward]).to("cpu")
        done = torch.Tensor([done]).to("cpu")
        action = torch.from_numpy(action).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        task = torch.Tensor(task).to("cpu")
        self.memory.add(state, reward, done, action, next_state, task)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states).to(self.device)
        rewards = torch.cat(batch.reward).view(self.batch_size, 1).to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        if self.out_actions:
            actions = torch.cat(batch.action).view(-1, self.out_actions).to(self.device)
        else:
            actions = torch.cat(batch.action).view(-1, self.n_actions).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states).to(self.device)
        tasks = torch.cat(batch.task).view(self.batch_size, self.task_dim).to(self.device)

        return states, rewards, dones, actions, next_states, tasks

    def train(self, episode, save):
        if len(self.memory) < self.batch_size:
            return 0, 0, 0
        else:
            batch = self.memory.sample(self.batch_size)
            states, rewards, dones, actions, next_states, tasks = self.unpack(batch)

            # Calculating the value target
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states, tasks)
            q1 = self.q_value_network1(states, reparam_actions, tasks)
            q2 = self.q_value_network2(states, reparam_actions, tasks)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.alpha * log_probs.detach()

            value = self.value_network(states, tasks)
            value_loss = self.value_loss(value, target_value)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = self.reward_scale * rewards + \
                           self.gamma * self.value_target_network(next_states, tasks) * (1 - dones)
            q1 = self.q_value_network1(states, actions, tasks)
            q2 = self.q_value_network2(states, actions, tasks)
            q1_loss = self.q_value_loss(q1, target_q)
            q2_loss = self.q_value_loss(q2, target_q)

            policy_loss = (self.alpha * log_probs - q).mean()
            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()


            self.soft_update_target_network(self.value_network, self.value_target_network)

            return value_loss.item(), 0.5 * (q1_loss + q2_loss).item(), policy_loss.item()

    def choose_action(self, states, tasks, torch=False, max_action=False, sigmoid=False):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        tasks = np.expand_dims(tasks, axis=0)
        tasks = from_numpy(tasks).float().to(self.device)
        if max_action:
            action, _, m = self.policy_network.sample_or_likelihood(states, tasks, max_action, sigmoid=sigmoid)
            return action.detach(), m

        action, _ = self.policy_network.sample_or_likelihood(states, tasks, max_action, sigmoid=sigmoid)
        if torch:
            return action.detach()
        return action.detach().cpu().numpy()[0]

    @staticmethod
    def soft_update_target_network(local_network, target_network, tau=0.005):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def set_to_eval_mode(self):
        self.policy_network.eval()
