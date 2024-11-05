# model/actor_critic.py
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.5, device="cpu"):
        super(ActorCritic, self).__init__()
        self.device = device
        self.state_dim = state_dim
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.action_var = torch.full((action_dim,), action_std_init**2).to(self.device)

    def act(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        if state.shape[1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, but got {state.shape[1]}")
            
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag(self.action_var))
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach(), self.critic(state).detach()

    def evaluate(self, state, action):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag_embed(self.action_var.expand_as(action_mean)))
        action_logprobs = dist.log_prob(action)
        state_values = self.critic(state)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy
