from abc import ABC, abstractmethod
from utils.types import *

class Env(ABC):
    '''
    Abstract Base Class for all environments currently supported.
    Environments are classes used to create a link between agents, models and
    the game. For cases where an environment for a game already exists, this class
    should still be used as a wrapper (eg. implementing an environment for openAI gym).
    '''

    def __init__(self, _id: str, render=False, reset_done=True, num_episodes=None):
        self.id = _id
        self.render = render
        self.reset_done = reset_done
        self.num_episodes = num_episodes if num_episodes else float('inf')

    
    # -> Type: ...
    # Forces the implemented methods to return a certain type
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def step(self, action):#action: Action) -> Tuple[Observation, Reward, Done]: ...
        pass

    @abstractmethod
    def reset(self): #-> Observation: ...
        pass

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def train(self, agent):
        pass

    @abstractmethod
    def play(self, agent, num_matches):
        pass
    