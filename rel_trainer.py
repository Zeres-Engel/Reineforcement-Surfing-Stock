from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def train_rl_model(env):
    """
    Huấn luyện mô hình RL sử dụng PPO.
    
    :param env: Môi trường RL.
    :return: Mô hình đã huấn luyện.
    """
    # Kiểm tra môi trường
    check_env(env, warn=True)
    
    # Khởi tạo mô hình PPO
    model = PPO('MlpPolicy', env, verbose=1)
    
    # Huấn luyện mô hình
    model.learn(total_timesteps=10000)
    
    return model
