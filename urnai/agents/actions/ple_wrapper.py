import random
import numpy as np
from .base.abwrapper import ActionWrapper
from utils.agent_utils import one_hot_encode, transformDistance, transformLocation


class PLEWrapper(ActionWrapper):

    def __init__(self, env):
        self.move_number = 0
        self.actions = env.env_instance.getActionSet()
        self.action_indices = [idx for idx in range(len(self.actions))]


    def is_action_done(self):
        return True

    
    def reset(self):
        self.move_number = 0


    def get_actions(self):
        return self.action_indices


    def get_excluded_actions(self, obs):        
        return []


    def get_action(self, action_idx, obs):
        return self.actions[action_idx]
