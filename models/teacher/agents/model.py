import torch
import torch.nn as nn

class Exp(nn.Module):


    def __init__(self):
        super().__init__()

    def forward(x):
        return torch.exp(x)

""" class TrendNet(nn.Module):

    def __init__(self, input_shape, action_shape):
        super().__init__()
        state_dim = input_shape[0]
        action_dim = action_shape[0]
        hidden_size_1 = 128
        hidden_size = 16

        self.shared_w = nn.Sequential(
            nn.Linear(state_dim, hidden_size_1),
            nn.Linear(hidden_size_1, hidden_size),
            nn.ReLU()
        )
        self.mean = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        self.std = nn.Sequential(
            nn.Linear(hidden_size, action_dim)
        )

        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        l = self.shared_w(x)
        mean = self.mean(l)
        std = torch.exp(self.std(l))
        dist = torch.distributions.Normal(mean, std)
        value = self.value(l) 
        return dist, value """
    


class TrendNet(nn.Module):

    def __init__(self, input_shape, action_shape):
        super().__init__()
        state_dim = input_shape[0] * input_shape[1]
        action_dim = action_shape[0]
        hidden_size_1 = 128
        hidden_size_2 = 64
        hidden_size = 16

        self.shared_w = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(state_dim, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size),
            nn.ReLU()
        )
        self.mean = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        self.std = nn.Sequential(
            nn.Linear(hidden_size, action_dim)
        )

        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        l = self.shared_w(x)
        mean = self.mean(l)
        std = torch.clamp(torch.exp(self.std(l)), 1e-6, 50)
        dist = torch.distributions.Normal(mean, std)
        value = self.value(l) 
        return dist, value
    