from abc import abstractmethod
from models.base.abmodel import LearningModel
from .abagent import Agent
from ..actions.base.abwrapper import ActionWrapper
from pysc2.lib import actions

class SC2Agent(Agent):
    def __init__(self, action_wrapper: ActionWrapper):
        super(SC2Agent, self).__init__(action_wrapper)
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.episodes += 1

    def step(self, obs):
        super(SC2Agent, self).step(obs)
        self.steps += 1
        self.reward += obs.reward
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])