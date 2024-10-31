from .test_agent import *
# external
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import torch.optim as optim
import pandas as pd
torch.autograd.set_detect_anomaly(True)

class PPO:
    
    def __init__(self, device, state_dim, action_dim):
        self.episodes = 50
        self.episode_length = 800

        ##hyperparameters
        self.critic_coeff = 0.5
        self.entropy_coeff = 0.01
        self.eps_clip = 0.2              # clip parameter for PPO
        self.gamma = 0.95                # discount factor
        self.batch_size = self.episode_length  #max_ep_len * 4      # update policy every n timesteps
        self.lr_actor = 0.0003       # learning rate for actor network
        self.lr_critic = 0.001       # learning rate for critic network
        self.K_epochs = 20 #40              # update policy for K epochs
        self.action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
        self.action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        self.min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
        self.action_std_decay_freq = int(30) 
        
        self.device = device
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, self.action_std, self.device).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, self.action_std, self.device).to(self.device)
        self.MseLoss = nn.MSELoss()

        self.train_specs = f'ccoeff{self.critic_coeff}-ecoeff{self.entropy_coeff}-eps_clip{self.eps_clip}-g{self.gamma}-b{self.batch_size}-lr_actor{self.lr_actor}-lr_critic{self.lr_critic}-K_eps{self.K_epochs}-a_std{self.action_std}-a_std_drate{self.action_std_decay_rate}-min_a_std{self.min_action_std}-a_std_dfreq{self.action_std_decay_freq}/'
        self.base = './testing/'
        self.chkpts_path = self.base + 'checkpoints/' + self.train_specs
        self.logs_path = self.base + 'logs/' + self.train_specs
        self.save_freq = 30

    def __load(self):
        chkpt_files = os.listdir(self.chkpts_path)
        latest_chkpt_file = 'latest.chkpt'
        best_chkpt_file = 'best.chkpt' 
        if (latest_chkpt_file in chkpt_files) and (best_chkpt_file in chkpt_files):
            latest_chkpt_dict = torch.load(self.chkpts_path + latest_chkpt_file)
            best_chkpt_dict = torch.load(self.chkpts_path + best_chkpt_file)

            self.global_step = latest_chkpt_dict['step'] + 1
            self.start_episode = latest_chkpt_dict['episode'] + 1
            self.end_episode = self.start_episode + self.episodes
            self.latest_reward = latest_chkpt_dict['reward']

            self.action_std = latest_chkpt_dict['action_std']
            self.policy.load_state_dict(latest_chkpt_dict['policy'])
            self.optimizer.load_state_dict(latest_chkpt_dict['optimizer'])
            """ for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr """

            self.best_reward = best_chkpt_dict['reward']
            self.best_policy_w = best_chkpt_dict['policy']
            self.best_optimizer_w = best_chkpt_dict['optimizer']
        else:
            self.global_step = 0
            self.start_episode = 0
            self.end_episode = self.episodes
            self.latest_reward = np.NINF

            self.best_reward = np.NINF
            self.best_policy_w = self.policy.state_dict()
            self.best_optimizer_w = self.optimizer.state_dict()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)
        
    def decay_action_std(self):
        print("--------------------------------------------------------------------------------------------")
        self.action_std = self.action_std - self.action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= self.min_action_std):
            self.action_std = self.min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()
    
    def select_action_test(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = self.policy_old.act_test(state)
        
        return action.detach().cpu().numpy().flatten()


    def update(self, next_state, terminated):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        if terminated:
          discounted_reward = 0
        else:
          with torch.no_grad():
            next_state = torch.FloatTensor(next_state).to(self.device)
            discounted_reward = self.policy_old.critic(next_state).item()
        for reward in reversed(self.buffer.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward) 
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
    
        # Optimize policy for K epochs
        for k in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2) 
            critic_loss = self.critic_coeff * self.MseLoss(state_values, rewards)
            entropy_loss = - self.entropy_coeff * dist_entropy
            loss = actor_loss + critic_loss + entropy_loss
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            if k == (self.K_epochs - 1):
                self.writer.add_scalar('Train/actor_loss', actor_loss.mean().item(), self.global_step)
                self.writer.add_scalar('Train/critic_loss', actor_loss.mean().item(), self.global_step)
                self.writer.add_scalar('Train/entropy_loss', actor_loss.mean().item(), self.global_step)
                self.writer.flush()  

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()

    def __init_train(self, env_specs=''):
        self.chkpts_path += env_specs
        self.logs_path += env_specs
        os.makedirs(self.chkpts_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        self.writer = SummaryWriter(self.logs_path)
        self.__load()

    def __train_one_episode(self, env, episode, seed):
        state, _ = env.reset(seed=seed) 
        current_ep_reward = 0

        for step in range(0, self.episode_length):
            state = state if type(state).__module__ == np.__name__ else state.__array__() #return ndarray

            action = self.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            self.buffer.rewards.append(reward)
            self.buffer.is_terminals.append(terminated)

            """ self.writer.add_scalar('Train/action', action, self.global_step)
            self.writer.add_scalar('Train/reward', reward, self.global_step)   
            self.writer.flush()   """    
            if terminated or ((step + 1) % self.batch_size == 0) :
                self.update(next_state, terminated)

            self.global_step += 1
            current_ep_reward += reward
            state = next_state

            if terminated:  break

        self.writer.add_scalar('Train/reward', current_ep_reward, episode)
        self.writer.flush()
        self.latest_reward = current_ep_reward
        if episode % self.save_freq == 0:
            torch.save( dict(episode=episode, step=(self.global_step -1), policy=self.policy_old.state_dict(),\
                            optimizer=self.optimizer.state_dict(), reward=current_ep_reward, action_std=self.action_std),
                        self.chkpts_path + f'{episode}.chkpt')


    def __test_one_episode(self, test_env, episode, seed):
        state, _ = test_env.reset(seed=seed)
        current_ep_reward = 0
        for step in range(self.episode_length):
            state = state if type(state).__module__ == np.__name__ else state.__array__() #return ndarray

            action = self.select_action_test(state)
            next_state, reward, terminated, truncated, info = test_env.step(action)

            current_ep_reward += reward
            state = next_state
            if terminated:  break
        
        self.writer.add_scalar('Test/reward', current_ep_reward, episode)  
        self.writer.flush()
        if current_ep_reward >= self.best_reward:
            self.best_reward = current_ep_reward
            self.best_policy_w = self.policy.state_dict()
            self.best_optimizer_w = self.optimizer.state_dict()

    def train(self, env, test_env, seed, env_specs):
        self.__init_train(env_specs)
        for episode in range(self.start_episode, self.end_episode):
            print(f'Episode {episode}/{self.end_episode - 1}')
            self.__train_one_episode(env, episode, seed)
            self.__test_one_episode(test_env, episode, seed )

            if ((episode + 1) % self.action_std_decay_freq == 0):
                self.decay_action_std()

        torch.save( dict(episode=(self.end_episode - 1), step=(self.global_step - 1), policy=self.best_policy_w,\
                        optimizer=self.best_optimizer_w, reward=self.best_reward, action_std=self.action_std),
                    self.chkpts_path + 'best.chkpt')
        torch.save( dict(episode=(self.end_episode - 1), step=(self.global_step - 1), policy=self.policy.state_dict(),\
                        optimizer=self.optimizer.state_dict(), reward=self.latest_reward, action_std=self.action_std),
                    self.chkpts_path + 'latest.chkpt')
        self.writer.close()

