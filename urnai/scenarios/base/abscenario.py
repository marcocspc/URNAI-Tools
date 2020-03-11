import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 

from abc import ABC, abstractmethod
from envs.base.abenv import ABEnv

class ABScenario(ABC, ABEnv):
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
