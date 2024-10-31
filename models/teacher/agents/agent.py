# -*- coding: utf-8 -*-

# standard library
import random
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
# external


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def len(self):
        return len(self.actions) 


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, device):
        super(ActorCritic, self).__init__()

        self.device = device
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        self.hidden_num = 64

        # actor
        self.actor = nn.Sequential(
                        nn.Flatten(start_dim=-2, end_dim=-1),
                        nn.Linear(state_dim, self.hidden_num),
                        nn.Tanh(),
                        nn.Dropout(0.2),
                        nn.Linear(self.hidden_num, self.hidden_num),
                        nn.Tanh(),
                        nn.Dropout(0.2),
                        nn.Linear(self.hidden_num, action_dim),
                        nn.Tanh()
                    )

        # critic
        self.critic = nn.Sequential(
                        nn.Flatten(start_dim=-2, end_dim=-1),
                        nn.Linear(state_dim, self.hidden_num),
                        nn.Tanh(),
                        nn.Dropout(0.2),
                        nn.Linear(self.hidden_num, self.hidden_num),
                        nn.Tanh(),
                        nn.Dropout(0.2),
                        nn.Linear(self.hidden_num, 1)
                    )
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)


    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        """ print(f'action_mean: {action_mean.shape}, cov_mat: {cov_mat.shape}, action: {action.shape}, action_logprob {action_logprob.shape}, state_val: {state_val.shape}')
        exit() """

        return action.detach(), action_logprob.detach(), state_val.detach(), action_mean.detach(), torch.sqrt(self.action_var).detach()
    
    def act_test(self, state):
        action = self.actor(state)
        return action.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
    
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        """ print(f'state {state.shape}, action {action.shape}')
        print(f'action_mean {action_mean.shape}, action_var {action_var.shape}, cov_mat {cov_mat.shape}')
        print(f' action_logprobs {action_logprobs.shape}, dist_entropy {dist_entropy.shape}, state_values {state_values.shape}')
        exit() """
        
        return action_logprobs, state_values, dist_entropy