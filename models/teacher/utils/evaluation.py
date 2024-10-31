import numpy as np
from utils.logger import get_logger



LOG = get_logger('Trainer')
def evaluate_poliy(model, env, seed, logger):
    state, _ = env.reset(seed=seed)
    env.render()
    current_ep_reward = 0
    step = 0
    logger.init_episode()
    model.policy_old.eval()
    model.policy.eval()
    while True:
        state = state if type(state).__module__ == np.__name__ else state.__array__() #return ndarray

        action = model.select_action_test(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        env.render()
        logger.log_step(info)

        step += 1
        current_ep_reward += (reward)
        state = next_state
        if terminated:  break
    
    accuracy, f1, cf_mt_dict = logger.record(1)
    LOG.info(f'Evaluation reward: {current_ep_reward}, Test Accuracy: {accuracy}, Test F1: {f1}')