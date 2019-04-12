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


    def reset(self):
        self.episodes += 1
        self.previous_action = None
        self.previous_state = None
        self.action_wrapper.reset()


    def learn(self, obs, reward, done):
        current_state = self.build_state(obs)
        if self.previous_state is not None:
            reward = self.get_reward(obs, reward, done)
            self.model.learn(self.previous_state, self.previous_action, reward, current_state, done)
        self.previous_state = current_state


    def play(self, obs):
        if self.action_wrapper.is_action_done():
            current_state = self.build_state(obs)
            predicted_action = self.model.predict(current_state)
            self.previous_action = predicted_action
        return [self.action_wrapper.get_action(self.previous_action, obs)]


    def step(self, obs, reward, done):
        self.steps += 1
        self.reward += reward

        if done:
            self.reset()
            return [actions.FUNCTIONS.no_op()]

        # Taking the first step for a smart action
        if self.action_wrapper.is_action_done():
            ## Building our agent's state
            current_state = self.build_state(obs)

            excluded_actions = self.action_wrapper.get_excluded_actions(obs)
            rl_action = self.model.choose_action(current_state, excluded_actions)

            self.previous_action = rl_action
        return [self.action_wrapper.get_action(self.previous_action, obs)]