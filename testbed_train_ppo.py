import os
import torch
import numpy as np
from rocket_env import RocketEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    task = 'hover'   # 'hover' or 'landing'
    task = 'landing' # 'hover' or 'landing'

    max_m_episode = 800_000
    max_steps = 800



    env = RocketEnv(task=task, max_steps=max_steps)
    env = DummyVecEnv([lambda: env])
    #print('CHECK_ENV', 'OK' if check_env(env) is None else 'ERROR')

    set_random_seed(0)

    model = PPO(
        'MlpPolicy',
        env,
        device=device,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        gamma=0.99
    )

    model.learn(total_timesteps=max_m_episode * max_steps, log_interval=100)

    model.save("ppo_model")

if __name__ == '__main__':
    main()