from abc import ABC, abstractmethod

class ActionWrapper(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def is_action_done(self) -> bool: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def get_actions(self):
        pass

    @abstractmethod
    def get_excluded_actions(self, obs):
        pass

    @abstractmethod
    def get_action_space_dim(self):
        return len(self.get_actions())

    @abstractmethod
    def get_action(self, one_hot_action, obs):
        pass

    
    