from abc import ABC, abstractmethod
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder
from utils.types import ActionIndex

class LearningModel(ABC):

    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, gamma, learning_rate, save_path, name=None):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.name = name
        self.action_wrapper = action_wrapper
        self.state_builder = state_builder
        self.actions = action_wrapper.get_actions()
        self.action_size = action_wrapper.get_action_space_dim()
        self.state_size = state_builder.get_state_dim()

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


    @abstractmethod
    def save(self) -> None : ...


    @abstractmethod
    def load(self) -> None: ...