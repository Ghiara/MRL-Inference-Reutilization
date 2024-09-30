import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rlkit.torch.distributions import TanhNormal

def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)

class TransferFunction(nn.Module):
    #TODO: change such that only **kwargs is passed
    def __init__(self, obs_dim: int, act_simple_dim:int, hidden_sizes:list, act_complex_dim:int, pretrained:str = None, init_w=3e-3):
        super(TransferFunction, self).__init__()
        # Define a simple neural network architecture
        if pretrained is not None: 
            print(f'Load model {pretrained}')
            full_network = self.load_model(pretrained)
            self.network = full_network.network
            self.mean_layer = full_network.mean_layer
            self.log_std_layer = full_network.log_std_layer

        else:
            sizes = [obs_dim + act_simple_dim] + list(hidden_sizes)
            layers = []
            for in_size, out_size in zip(sizes[:-1], sizes[1:]):
                layers.append(torch.nn.Linear(in_size, out_size))
                init_weight(layers[-1])
                layers[-1].bias.data.zero_()
                layers.append(torch.nn.ReLU())
            self.network = torch.nn.Sequential(*layers)
            self.mean_layer = torch.nn.Linear(out_size, act_complex_dim)
            init_weight(self.mean_layer, initializer="xavier uniform")
            self.mean_layer.bias.data.zero_()
            self.log_std_layer = torch.nn.Linear(out_size, act_complex_dim)
            init_weight(self.log_std_layer, initializer="xavier uniform")
            self.log_std_layer.bias.data.zero_()

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        if action.dim() != observation.dim():
            action = action.unsqueeze(-1)
        if observation.dim() == 0:
            observation = observation.unsqueeze(-1)
        x = torch.concatenate([observation, action], dim=-1).to(torch.float32)
        x = self.network(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Typic
        return mean, log_std

    def get_action(self, observation: torch.Tensor, action: torch.Tensor, return_dist:bool = False):
        mean, log_std = self.forward(observation, action)

        if return_dist:
            tanh_normal = TanhNormal(mean.cpu(),torch.exp(log_std).cpu())
            z, log_prob  = tanh_normal.rsample_and_logprob()
            # log_prob= tanh_normal.log_prob(
            #         action,
            #         pre_tanh_value=pre_tanh_value
            #     )
            # log_prob = log_prob.sum(dim=1, keepdim=True)
            return z.cuda(), log_prob.cuda()

        return torch.tanh(mean)

    def train(self, batch, lr=1e-4):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        reward1, reward2 = batch['r_simple'], batch['r_complex'] # Get new rewards for each epoch
        optimizer.zero_grad()
        output = self.forward(reward1)
        loss = criterion(output, reward2)
        loss.backward()
        optimizer.step()

    def load_model(self, path):
        network = torch.load(path)
        return network


class RewardPredictor(nn.Module):
    #TODO: change such that only **kwargs is passed
    def __init__(self, obs_dim: int, act_complex_dim:int, hidden_sizes:list, task_dim:int=1, reward_dim:int=1, pretrained:str = None, init_w=3e-3):
        super(RewardPredictor, self).__init__()
        # Define a simple neural network architecture

        if pretrained is not None: 
            print(f'Load model {pretrained}')
            self.network = self.load_model(pretrained).network

        else:
            sizes = [obs_dim + act_complex_dim + task_dim] + list(hidden_sizes)
            layers = []
            for in_size, out_size in zip(sizes[:-1], sizes[1:]):
                layers.append(torch.nn.Linear(in_size, out_size))
                init_weight(layers[-1])
                layers[-1].bias.data.zero_()
                layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(out_size, reward_dim))
            init_weight(layers[-1], initializer="xavier uniform")
            layers[-1].bias.data.zero_()
            # layers.append(torch.nn.ReLU())
            self.network = torch.nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor, action: torch.Tensor, task: torch.Tensor):
        if task.dim() != observation.dim():
            task = task.unsqueeze(-1)
        x = torch.concatenate([observation, action, task], dim=-1).to(torch.float32)
        x = self.network(x)
        return x

    def train(self, batch, lr=1e-4):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        reward1, reward2 = batch['r_simple'], batch['r_complex'] # Get new rewards for each epoch
        optimizer.zero_grad()
        output = self.forward(reward1)
        loss = criterion(output, reward2)
        loss.backward()
        optimizer.step()

    def load_model(self, path):
        network = torch.load(path)
        return network


class ValueNetwork(nn.Module):
    #TODO: change such that only **kwargs is passed
    def __init__(self, obs_dim: int, hidden_sizes:list, task_dim:int=1, reward_dim:int=1, pretrained:str = None, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        # Define a simple neural network architecture

        if pretrained is not None: 
            print(f'Load model {pretrained}')
            self.network = self.load_model(pretrained).network

        else:
            sizes = [obs_dim + task_dim] + list(hidden_sizes)
            layers = []
            for in_size, out_size in zip(sizes[:-1], sizes[1:]):
                layers.append(torch.nn.Linear(in_size, out_size))
                init_weight(layers[-1])
                layers[-1].bias.data.zero_()
                layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(out_size, reward_dim))
            init_weight(layers[-1], initializer = "xavier uniform")
            layers[-1].bias.data.zero_()
            # layers.append(torch.nn.ReLU())
            self.network = torch.nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor, task: torch.Tensor):
        if task.dim() != observation.dim():
            task = task.unsqueeze(-1)
        x = torch.concatenate([observation, task], dim=-1).to(torch.float32)
        x = self.network(x)
        return x

    def train(self, batch, lr=1e-4):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # reward1, reward2 = batch['r_simple'], batch['r_complex'] # Get new rewards for each epoch
        # optimizer.zero_grad()
        # output = self.forward(reward1)
        # loss = criterion(output, reward2)
        # loss.backward()
        # optimizer.step()

    def load_model(self, path):
        network = torch.load(path)
        return network