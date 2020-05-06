import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from abc import ABC, abstractmethod
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder
from utils.returns import ActionIndex
from urnai.base.savable import Savable 

class LearningModel(Savable):

    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, gamma, learning_rate, epsilon_start, epsilon_min, epsilon_decay_rate, per_episode_epsilon_decay=False, name=None):
        super(LearningModel, self).__init__()
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.name = name
        self.action_wrapper = action_wrapper
        self.state_builder = state_builder
        self.actions = action_wrapper.get_actions()
        self.action_size = action_wrapper.get_action_space_dim()
        self.state_size = state_builder.get_state_dim()

        # EXPLORATION PARAMETERS FOR EPSILON GREEDY STRATEGY
        self.epsilon_greedy = epsilon_start 
        self.epsilon_min = epsilon_min 
        self.epsilon_decay_rate = epsilon_decay_rate 
        self.per_episode_epsilon_decay = per_episode_epsilon_decay

    @abstractmethod
    def learn(self, s, a, r, s_, done, is_last_step: bool) -> None : ...


    @abstractmethod
    def choose_action(self, state, excluded_actions=[]) -> ActionIndex:
        '''
        Implements the exploration exploitation method for the model.
        '''
        pass


    @abstractmethod
    def predict(self, state, excluded_actions=[]) -> ActionIndex:
        '''
        Given a State, returns the index for the action with the highest Q-Value.
        '''
        pass

    def decay_epsilon(self):
        if self.epsilon_greedy > self.epsilon_min:
            self.epsilon_greedy *= self.epsilon_decay_rate

    def ep_reset(self):
        if self.per_episode_epsilon_decay:
            self.decay_epsilon()
