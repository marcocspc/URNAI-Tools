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
    def get_reward(self, obs, reward, done):
        pass

    @abstractmethod
    def get_state_dim(self):
        pass

    def setup(self, env):
        '''
        All agents need to have a setup method because PySC2 agents require a setup method, so this is just boilerplate code
        required for the Trainer class to work, since the setup method must be called inside both play and train methods for PySC2 agents to work.
        '''
        i = 0

    @abstractmethod
    def step(self, obs, reward, done):
        pass
    
    @abstractmethod
    def play(self, obs):
        pass

    def set_model(self, model):
        self.model = model
