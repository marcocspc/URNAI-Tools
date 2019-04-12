from abc import ABC, abstractmethod
from agents.actions.base.abwrapper import ActionWrapper
from agents.base.abagent import Agent

class LearningModel(ABC):

    def __init__(self, agent: Agent, save_path, name=None):
        self.save_path = save_path
        self.name = name
        self.actions = agent.action_wrapper.get_actions()
        self.action_size = agent.action_wrapper.get_action_space_dim()
        self.state_size = agent.get_state_dim()
        agent.set_model(self)

    @abstractmethod
    def learn(self, s, a, r, s_, done):
        pass

    '''
    Implements the exploration exploitation method for the model.
    '''
    @abstractmethod
    def choose_action(self, state, excluded_actions=[], is_playing=False):
        pass

    '''
    Given a State, returns the action with the highest Q-Value.
    '''
    @abstractmethod
    def predict(self, state):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass