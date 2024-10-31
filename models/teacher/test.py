
import gymnasium as gym
import os
import time
from datetime import datetime
import torch
import random
from tests.test_trainer import *

################################## set device ##################################

print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")


################################### Training ###################################


####### initialize environment hyperparameters ######

env_name = "Pendulum-v1"

random_seed = 13        # set random seed if required (0 = no random seed)
env = gym.make(env_name)
test_env = gym.make(env_name)

# state space dimension
state_dim = env.observation_space.shape[0]

action_dim = env.action_space.shape[0]
if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

# initialize a PPO agent
ppo_agent = PPO(device, state_dim, action_dim)
ppo_agent.train(env, test_env, random_seed, '')


""" seed = 13
set_seed(seed)
timesteps_dim = 3
env = gym.make('Pendulum-v1', g=9.81)

state_shape = env.observation_space.shape
action_shape = env.action_space.shape


env = FrameStack(env, num_stack=timesteps_dim)

agent_state_shape = (timesteps_dim, ) + state_shape

env_fns = [
        lambda: FrameStack(env = gym.make('Pendulum-v1', g=9.81), num_stack=timesteps_dim),
        
]
test_env = gym.vector.SyncVectorEnv(env_fns)

print(state_shape)
print(action_shape)
print(agent_state_shape)
state, _ = env.reset()
print(state.__array__())
next_state, reward, terminated, truncated, info = env.step(np.array([-1]))
print(next_state.__array__())
print(reward)
print(terminated)
print(info)
print(truncated)

test_state, _ = test_env.reset()
print(test_state)


agent = TestAgent(agent_state_shape, action_shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = A2CTrainer(agent, device, env, test_env)
trainer.train() """