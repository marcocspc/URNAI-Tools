import random
import math
import os.path
import numpy as np
from .base.abagent import Agent
from pysc2.lib import actions, features
from pysc2.agents.base_agent import BaseAgent


class CartpoleAgent(Agent):
    
    def __init__(self, action_wrapper, env):
        super(CartpoleAgent, self).__init__(action_wrapper)
        self.state_dim = env.env_instance.observation_space.shape[0]
        

    def get_reward(self, obs, reward, done):
        return reward


    def build_state(self, obs):
        obs = obs.reshape(1, self.get_state_dim())
        #obs = np.expand_dims(obs, axis=0)
        return obs


    def get_state_dim(self):
        return self.state_dim
    
