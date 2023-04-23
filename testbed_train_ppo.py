import os
import torch
import numpy as np
from rocket_env_trial import Rocket
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_checkpoint_callback(local_vars, global_vars, max_steps):
    """
    Custom callback function to save model checkpoints.
    """
    model = local_vars['self']
    timestep = model.num_timesteps

    # Save a checkpoint every 1000 episodes
    if timestep % (1000 * max_steps) == 0:
        checkpoint_folder = 'landing_poo_ckpt'
        os.makedirs(checkpoint_folder, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_folder, f'ppo_checkpoint_{timestep}')
        model.save(checkpoint_path)
        print(f"Saved checkpoint at timestep {timestep}")


def main():
    task = 'hover'   # 'hover' or 'landing'
    task = 'landing' # 'hover' or 'landing'

    max_m_episode = 80_000
    max_steps = 800



    env = Rocket(task=task, max_steps=max_steps)
    env = DummyVecEnv([lambda: env])
    #print('CHECK_ENV', 'OK' if check_env(env) is None else 'ERROR')

    set_random_seed(0)

    # Load the last checkpoint if available
    checkpoint_folder = os.path.join('./', task+"_ppo" + '_ckpt')
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    checkpoints = [f for f in os.listdir(checkpoint_folder) if f.startswith('ppo_checkpoint_')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        model = PPO.load(os.path.join(checkpoint_folder, latest_checkpoint), env=env)
        print(f"Loaded the latest checkpoint: {latest_checkpoint}")
    else:
        model = PPO(
            'MlpPolicy',
            env,
            device=device,
            verbose=1,
            tensorboard_log="./ppo_tensorboard/",
            gamma=0.99
        )

    model.learn(total_timesteps=max_m_episode * max_steps, log_interval=100,
                callback=lambda local_vars, global_vars: save_checkpoint_callback(local_vars, global_vars, max_steps))

    model.save("ppo_model")

if __name__ == '__main__':
    main()