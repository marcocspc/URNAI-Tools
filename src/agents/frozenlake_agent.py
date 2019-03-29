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
        
        self.previous_action = None
        self.previous_state = None
        

    def get_reward(self, obs):
        if obs == 1:
            return 1000
        elif obs == 0:
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


    def play(self, obs, obs_reward, done):
        # Taking the first step for a smart action
        ## Building our agent's state
        current_state = self.build_state(obs)

        # Selects an action by passing the current state as a string. Our dataframe is indexed by strings.
        # We only select an action when move_number == 0, and we keep it until move_number == 2
        predicted_action = self.model.choose_action(current_state, is_playing=True)
        
        return self.action_wrapper.get_action(predicted_action, obs)


    def step(self, obs, obs_reward, done):
        # Taking the first step for a smart action
        if self.action_wrapper.is_action_done():
            ## Building our agent's state
            current_state = self.build_state(obs)
            
            # If it's not the first step, we can learn
            if self.previous_action is not None:
                reward = self.get_reward(obs_reward)

                self.model.learn(self.previous_state, self.previous_action, reward, current_state, done)


            excluded_actions = self.action_wrapper.get_excluded_actions(obs)

            # Selects an action by passing the current state as a string. Our dataframe is indexed by strings.
            # We only select an action when move_number == 0, and we keep it until move_number == 2
            rl_action = self.model.choose_action(current_state, excluded_actions)

            self.previous_state = current_state
            self.previous_action = rl_action

        return self.action_wrapper.get_action(self.previous_action, obs)
