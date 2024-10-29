from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def train_rl_model(env):
    check_env(env, warn=True)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    return model
