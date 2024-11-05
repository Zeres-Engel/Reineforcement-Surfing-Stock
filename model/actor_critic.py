# model/actor_critic.py
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.5, device="cpu"):
        super(ActorCritic, self).__init__()
        self.device = device
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.action_var = torch.full((action_dim,), action_std_init**2).to(self.device)

    def act(self, state):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag(self.action_var))
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach(), self.critic(state).detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag_embed(self.action_var.expand_as(action_mean)))
        action_logprobs = dist.log_prob(action)
        state_values = self.critic(state)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy
