""" main.py """

from configs.config import CFG
from utils.config import Config
from data_loaders.vn30_loader import DataLoader
from environments.vn30_trend_env import VN30TrendEnv
from executors.ppo_trainer import *
from utils.metrics_logger import *
from utils.evaluation import evaluate_poliy
from gymnasium.wrappers import FrameStack
import gymnasium
import torch
import random
import numpy as np
from gymnasium.wrappers.record_video import RecordVideo

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def run():
    seed = 13
    set_seed(seed)
    config = Config.from_json(CFG)
    data_loader = DataLoader.from_json(config.data)
    vn30_tickets = data_loader.get_tickets()
    trading_days = data_loader.get_trading_days()
    timesteps_dim = data_loader.get_timesteps_dim()
    features_dim = data_loader.get_features_dim()

    state_dim = timesteps_dim * features_dim
    action_dim = 1

    mode = config.agent.mode
    
    ticket = data_loader.get_ticket()
    env = VN30TrendEnv(data_loader, features_dim, action_dim, config.agent.env, seed, ticket=ticket, mode='train') 
    env = FrameStack(env, num_stack=timesteps_dim)
    info_keys = env.get_info_keys()  
    env_fns = [
                lambda: FrameStack(VN30TrendEnv(data_loader, features_dim, action_dim, config.agent.env, seed, ticket=ticket, mode='validation'), num_stack=timesteps_dim),
                
    ]

    test_env = gymnasium.vector.SyncVectorEnv(env_fns)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = TrainLogger(info_keys, vn30_tickets, trading_days)
    test_logger = TestLogger(info_keys, vn30_tickets, trading_days)

    model = PPO(device, state_dim, action_dim, config.agent.train, config.agent.save, logger, test_logger)
    model.train(env, test_env, seed, config.agent.env, ticket)
    if mode == "evaluation":
        model.load_best_model()
        env_specs = f'shift_reward{config.agent.env.shift_reward}-scale_reward{config.agent.env.scale_reward}/' + f'{ticket}/'
        video_path = model.get_specs_path() + env_specs + "./video/"
        
        eval_env = VN30TrendEnv(data_loader, features_dim, action_dim, config.agent.env, seed, ticket=ticket, mode='train', render_mode="rgb_array")
        eval_env = FrameStack(RecordVideo(eval_env, video_folder=video_path, video_length=205, name_prefix="rendered_env_video", disable_logger=False), num_stack=timesteps_dim)
        eval_logger = TrainLogger(info_keys, vn30_tickets, trading_days)
        evaluate_poliy(model, eval_env, seed, eval_logger)
    

    """ print(vn30_tickets)
    state, _ = env.reset(options={'mode' : 'episode', 'episode' : 100})
    print(state.__array__())
    next_state, reward, terminated, truncated, info = env.step(np.array([-1]))
    print(next_state.__array__())
    print(reward)
    print(terminated)
    print(info)
    print(truncated)
    
    print(test_env.observation_space)
    print(test_env.action_space)

    state, info = test_env.reset()
    print(state.shape)
    print(info)
    next_state, reward, terminated, truncated, info = test_env.step(np.ones((30, 1), dtype='float32'))
    print(next_state.shape)
    print(reward.shape)
    print(terminated.shape)
    print(info)
    print(truncated.shape) """
    
if __name__ == "__main__":
    run()
