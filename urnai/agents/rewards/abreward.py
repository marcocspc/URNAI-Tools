from abc import ABC, abstractmethod
from agents.actions.base.abwrapper import ActionWrapper
from utils.returns import Reward

class RewardBuilder(ABC):
    '''
    Every Agent needs to own an instance of this base class in order to calculate its rewards. So every time we want to create a new agent,
    we should either use an existing RewardBase implementation or create a new one.
    '''
    @abstractmethod
    def get_reward(self, obs) -> Reward: ...
