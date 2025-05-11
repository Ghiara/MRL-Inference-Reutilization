import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class ValueNetwork(nn.Module):
    def __init__(self, n_states, task_dim=1, pretrained=None, hidden_layers=[256,256]):
        super(ValueNetwork, self).__init__()

        if pretrained:
            self.n_states = n_states
            print(f'Load model {pretrained}')
            self.network = self.load_model(pretrained)
            for layer_name, layer in self.network.named_children():
                setattr(self, layer_name, layer)
            delattr(self,'network')

        else:
            self.n_states = n_states
            self.n_hidden_filters = hidden_layers[0]
            sizes = [self.n_states + task_dim] + list(hidden_layers)
            i = 1
            for in_size, out_size in zip(sizes[:-1], sizes[1:]):
                layer_name = f'hidden{i}'
                layer = nn.Linear(in_features=in_size, out_features=out_size)
                init_weight(layer)
                layer.bias.data.zero_()
                setattr(self, layer_name, layer)
                i+=1
            # self.hidden1 = nn.Linear(in_features=self.n_states + task_dim, out_features=self.n_hidden_filters)
            # init_weight(self.hidden1)
            # self.hidden1.bias.data.zero_()
            # self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
            # init_weight(self.hidden2)
            # self.hidden2.bias.data.zero_()
            # self.hidden3 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
            # init_weight(self.hidden3)
            # self.hidden3.bias.data.zero_()
            # self.hidden4 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
            # init_weight(self.hidden4)
            # self.hidden4.bias.data.zero_()
            self.value = nn.Linear(in_features=sizes[-1], out_features=1)
            init_weight(self.value, initializer="xavier uniform")
            self.value.bias.data.zero_()

    def forward(self, states, tasks):
        x = torch.cat([states, tasks], dim=1)
        for layer_name, layer in self.named_children():
                if 'hidden' in layer_name:
                    x = F.relu(layer(x))
        return self.value(x)
    
    def load_model(self, path):
        network = torch.load(path)
        return network


class QvalueNetwork(nn.Module):
    def __init__(self, n_states, n_actions, task_dim=1, n_hidden_filters=256, pretrained=None, hidden_layers=[256,256]):
        super(QvalueNetwork, self).__init__()

        if pretrained:
            self.n_states = n_states
            self.n_actions = n_actions
            print(f'Load model {pretrained}')
            self.network = self.load_model(pretrained)
            for layer_name, layer in self.network.named_children():
                setattr(self, layer_name, layer)
            delattr(self,'network')

        else:
            self.n_states = n_states
            self.n_hidden_filters = n_hidden_filters
            self.n_actions = n_actions
            sizes = [self.n_states + self.n_actions + task_dim] + list(hidden_layers)
            i = 1
            for in_size, out_size in zip(sizes[:-1], sizes[1:]):
                layer_name = f'hidden{i}'
                layer = nn.Linear(in_features=in_size, out_features=out_size)
                init_weight(layer)
                layer.bias.data.zero_()
                setattr(self, layer_name, layer)
                i+=1

            # self.hidden1 = nn.Linear(in_features=self.n_states + self.n_actions + task_dim, out_features=self.n_hidden_filters)
            # init_weight(self.hidden1)
            # self.hidden1.bias.data.zero_()
            # self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
            # init_weight(self.hidden2)
            # self.hidden2.bias.data.zero_()
            # self.hidden3 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
            # init_weight(self.hidden3)
            # self.hidden3.bias.data.zero_()
            # self.hidden4 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
            # init_weight(self.hidden4)
            # self.hidden4.bias.data.zero_()
            self.q_value = nn.Linear(in_features=sizes[-1], out_features=1)
            init_weight(self.q_value, initializer="xavier uniform")
            self.q_value.bias.data.zero_()

    def forward(self, states, actions, tasks):
        x = torch.cat([states, actions, tasks], dim=1)
        for layer_name, layer in self.named_children():
            if 'hidden' in layer_name:
                x = F.relu(layer(x))
        return self.q_value(x)
    
    def load_model(self, path):
        network = torch.load(path)
        return network


class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions, action_bounds, task_dim=1, n_hidden_filters=256, pretrained=None, hidden_layers=[256,256], out_actions=None):
        super(PolicyNetwork, self).__init__()
        if pretrained:
            self.n_states = n_states
            self.n_actions = n_actions
            self.action_bounds = action_bounds
            print(f'Load model {pretrained}')
            self.network = self.load_model(pretrained)
            for layer_name, layer in self.network.named_children():
                setattr(self, layer_name, layer)

            delattr(self,'network')

        else:
            self.n_states = n_states
            self.n_hidden_filters = n_hidden_filters
            self.n_actions = n_actions
            self.action_bounds = action_bounds
            sizes = [self.n_states + task_dim] + list(hidden_layers)
            i = 1
            for in_size, out_size in zip(sizes[:-1], sizes[1:]):
                layer_name = f'hidden{i}'
                layer = nn.Linear(in_features=in_size, out_features=out_size)
                init_weight(layer)
                layer.bias.data.zero_()
                setattr(self, layer_name, layer)
                i+=1

            # self.hidden1 = nn.Linear(in_features=self.n_states + task_dim, out_features=self.n_hidden_filters)
            # init_weight(self.hidden1)
            # self.hidden1.bias.data.zero_()
            # self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
            # init_weight(self.hidden2)
            # self.hidden2.bias.data.zero_()
            # self.hidden3 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
            # init_weight(self.hidden3)
            # self.hidden3.bias.data.zero_()
            # self.hidden4 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
            # init_weight(self.hidden4)
            # self.hidden4.bias.data.zero_()
            if out_actions:
                self.n_actions = out_actions
            self.mu = nn.Linear(in_features=sizes[-1], out_features=self.n_actions)
            init_weight(self.mu, initializer="xavier uniform")
            self.mu.bias.data.zero_()

            self.log_std = nn.Linear(in_features=sizes[-1], out_features=self.n_actions)
            init_weight(self.log_std, initializer="xavier uniform")
            self.log_std.bias.data.zero_()

        self.action_selector = nn.Sequential(
            nn.Linear(list(self.children())[-4].out_features + task_dim, 300),
            nn.ReLU(),  # Non-linearity
            nn.Linear(300, 300),
            nn.ReLU(),  # Non-linearity
            nn.Linear(300, self.n_actions)
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, states, tasks, max_action=False, sigmoid=False):
        if states.dtype == torch.float64:
            states = states.to(torch.float32)
        if tasks.dtype == torch.float64:
            tasks = tasks.to(torch.float32)
        x = torch.cat([states, tasks], dim=-1)
        for layer_name, layer in self.named_children():
            if 'hidden' in layer_name:
                x = F.relu(layer(x))
                self.x = x
        mu = self.mu(x)
        if sigmoid:
            return self.tanh(mu)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist

    def sample_or_likelihood(self, states, tasks, max_action=False, sigmoid=False):
        if sigmoid:
            if max_action:
                dist = self(states, tasks, max_action, sigmoid=True)
                action_selector_input = torch.cat((self.x, tasks), dim=-1)  # Concatenate self.x and tasks
                m = torch.argmax(nn.functional.softmax(self.action_selector(action_selector_input), dim=-1))
                return dist, m
            return self(states, tasks, max_action, sigmoid=True), None
        dist = self(states, tasks, max_action)
        # Reparameterization trick
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        # return (action * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1]), log_prob
        if max_action:
            return action.clamp(self.action_bounds[0], self.action_bounds[1]), log_prob, m
        return action.clamp(self.action_bounds[0], self.action_bounds[1]), log_prob
    
    def get_action(self, states, tasks, return_dist=False):
        return self.sample_or_likelihood(states, tasks)[0]
    
    def load_model(self, path):
        network = torch.load(path)
        return network
