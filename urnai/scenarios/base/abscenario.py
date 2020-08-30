from abc import abstractmethod
from urnai.envs.base.abenv import Env
from urnai.agents.rewards.abreward import RewardBuilder
from urnai.agents.actions.base.abwrapper import ActionWrapper 
from urnai.agents.states.abstate import StateBuilder 

class ABScenario(Env):
    '''
        Abstract class for all Scenarios supported. A Scenario works like an environment, but uses it
        as a base to create a trainment pattern. For example, for an environment like DeepRTS, one can train an agent
        to learn how to collect specific resources, on a specific map.

        A scenario also offers a default RewardBuilder and ActionWrapper to help build an agent.
    '''

    '''
        Returns the default RewardBuilder
    '''
    @abstractmethod
    def get_default_reward_builder(self) -> RewardBuilder: ...

    '''
        Returns the default ActionWrapper 
    '''
    @abstractmethod
    def get_default_action_wrapper(self) -> ActionWrapper: ...

    '''
        Returns the default StateBuilder 
    '''
    @abstractmethod
    def get_default_state_builder(self) -> StateBuilder: ...
