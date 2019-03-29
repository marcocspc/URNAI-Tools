import random
import numpy as np
from .base.abwrapper import ActionWrapper
from utils.agent_utils import one_hot_encode_smart_actions, transformDistance, transformLocation


class GymWrapper(ActionWrapper):

    def __init__(self, env):
        self.move_number = 0

        self.actions = [action_idx for action_idx in range(env.env_instance.action_space.n)]   
        self.encoded_actions = one_hot_encode_smart_actions(self.actions)
        self.action_space_dim = env.env_instance.action_space.n


    def is_action_done(self):
        return True

    
    def reset(self):
        self.move_number = 0


    def get_actions(self):
        return self.encoded_actions


    def get_action_space_dim(self):
        return self.action_space_dim


    def get_excluded_actions(self, obs):        
        return []


    def get_action(self, one_hot_action, obs):
        smart_action = self.actions[np.argmax(one_hot_action)]
        return smart_action
