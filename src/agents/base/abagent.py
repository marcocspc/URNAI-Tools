from abc import ABC, abstractmethod
from agents.actions.base.abwrapper import ActionWrapper

class Agent(ABC):
    
    def __init__(self, action_wrapper: ActionWrapper):
        self.model = None
        self.action_wrapper = action_wrapper

    @abstractmethod
    def build_state(self, obs):
        pass

    @abstractmethod
    def get_state_dim(self):
        pass

    @abstractmethod
    def step(self, obs):
        pass

    def set_model(self, model):
        self.model = model
