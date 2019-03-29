from abc import ABC, abstractmethod


class ActionWrapper(ABC):
    """
    ActionWrapper works as an extra abstraction layer used by the agent to select actions. This means the agent doesn't select actions from action_set,
    but from ActionWrapper. This class is responsible to telling the agents which actions it can use and which ones are excluded from selection. It can
    also force the agent to use certain actions by combining them into multiple steps
    """

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

    
    