import gym
from gym import spaces
import numpy as np
from rocket import Rocket

import gym
from gym import spaces
import numpy as np

# Assuming you've already imported the Rocket class from the provided link

class RocketEnv(gym.Env):
    def __init__(self, max_steps,task,rocket_type, viewport_h , path_to_bg_img):
        self.task= task
        self.rocket_type=rocket_type
        self.viewport_h= viewport_h
        self.path_to_bg_img= path_to_bg_img
        self.max_steps = max_steps
        # Set up the action space
        self.action_space = spaces.Discrete(2)
        # Set up the observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        # Set up the reward range
        self.reward_range = (-np.inf, np.inf)
        # Set up the metadata
        self.metadata = {}


        super(Rocket, object).__init__(max_steps, task=self.task, rocket_type=self.rocket_type, viewport_h=self.viewport_h, path_to_bg_img=self.path_to_bg_img)

    def reset(self):
        return self.reset()

    def step(self, action):
        obs, reward, done, _ = self.step(action)
        return obs, reward, done, {}

    def render(self, mode='human'):
        self.render()