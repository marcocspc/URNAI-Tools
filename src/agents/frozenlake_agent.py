import random
import math
import os.path
import numpy as np
from .base.abagent import Agent
from pysc2.lib import actions, features
from pysc2.agents.base_agent import BaseAgent


class FrozenLakeAgent(Agent):
    
    def __init__(self, action_wrapper):
        super(FrozenLakeAgent, self).__init__(action_wrapper)
        

    def get_reward(self, obs, reward, done):
        if reward == 1:
            return 1000
        elif reward == 0:
            return 1
        else:
            return -1000


    def build_state(self, obs):
        if obs != None:
            index = obs
            obs = np.zeros((1, 16))
            obs[0][index] = 1
            return obs
        else:
            return None


    def get_state_dim(self):
        return 16
