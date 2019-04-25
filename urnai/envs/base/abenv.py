from abc import ABC, abstractmethod
from utils.types import *

class Env(ABC):
    '''
    Abstract Base Class for all environments currently supported.
    Environments are classes used to create a link between agents, models and
    the game. For cases where an environment for a game already exists, this class
    should still be used as a wrapper (e.g. implementing an environment for OpenAI gym).
    '''
    def __init__(self, _id: str, render=False, reset_done=True):
        self.id = _id
        self.render = render
        self.reset_done = reset_done
        self.env_instance = None

    '''
    Starts the environment. The implementation should assign the value of env_instance.
    '''
    @abstractmethod
    def start(self) -> None: ...

    '''
    Executes an action on the environment and returns an [Observation, Reward, Done] tuple.
    '''
    @abstractmethod
    def step(self, action) -> Tuple[Observation, Reward, Done]: ...


    '''
    Resets the environment. This method should return an Observation, since it's used by the Trainer to get the first Observation.
    '''
    @abstractmethod
    def reset(self) -> Observation: ...


    '''
    Closes the environment.
    '''
    @abstractmethod
    def close(self) -> None: ...
    