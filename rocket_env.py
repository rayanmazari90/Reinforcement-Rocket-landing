"""
+--------------------------------------------------------------------------------+
|  WARNING!!!                                                                    |
|  THIS IS JUST AN STUB FILE (TEMPLATE)                                          |
|  PROBABLY ALL LINES SHOULD BE CHANGED OR TOTALLY REPLACED IN ORDER TO GET A    |
|  WORKING FUNCTIONAL VERSION FOR YOUR ASSIGNMENT                                |
+--------------------------------------------------------------------------------+
"""

import gym
from gym import spaces
import numpy as np
from rocket import Rocket

class RocketEnv(gym.Env):
    def __init__(self):
        super(RocketEnv, self).__init__()
        
        # Set up action and observation spaces
        self.action_space = spaces.Discrete(2) # Assuming 2 actions: 0 - no thrust, 1 - thrust
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(4,), dtype=float)
        
        # Initialize the Rocket object
        self.rocket = Rocket()

    def step(self, action):
        # Execute one time step within the environment
        self.rocket.update(action)

        # Calculate reward and check if done
        reward = self._calculate_reward()
        done = self._is_done()

        return self.rocket.state(), reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.rocket.reset()
        return self.rocket.state()

    def render(self, mode='human'):
        # Render the environment to the screen
        pass

    def _calculate_reward(self):
        # Calculate the reward based on the current state
        if self.rocket.landed():
            return 100
        elif self.rocket.crashed():
            return -100
        else:
            return -1

    def _is_done(self):
        # Check if the episode is finished based on the current state
        return self.rocket.landed() or self.rocket.crashed()

if __name__ == "__main__":
    
    env = RocketEnv()
    print('CHECK_ENV', 'OK' if check_env(env) is None else 'ERROR')
    print(env.observation_space)
    print(env.action_space)
    print(type(env).__name__)