from abc import abstractmethod
from models.base.abmodel import LearningModel
from .abagent import Agent
from ..actions.base.abwrapper import ActionWrapper
from pysc2.lib import actions, features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_SELF = 1

class SC2Agent(Agent):
    def __init__(self, action_wrapper: ActionWrapper):
        super(SC2Agent, self).__init__(action_wrapper)
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None
        self.previous_action = None


    def setup(self, env):
        self.obs_spec = env.env_instance.observation_spec()
        self.action_spec = env.env_instance.action_spec()


    # def setup(self, obs_spec, action_spec):
    #     self.obs_spec = obs_spec
    #     self.action_spec = action_spec


    def reset(self):
        self.episodes += 1
        self.previous_action = None
        self.previous_state = None
        self.action_wrapper.reset()


    def play(self, obs):
        if self.action_wrapper.is_action_done():
            current_state = self.build_state(obs)
            predicted_action = self.model.choose_action(current_state, is_playing=True)
            self.previous_action = predicted_action
        return [self.action_wrapper.get_action(self.previous_action, obs)]


    def step(self, obs, reward, done):
        super(SC2Agent, self).step(obs, reward, done)
        self.steps += 1
        self.reward += reward
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])