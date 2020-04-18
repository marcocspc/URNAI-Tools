from abc import abstractmethod
from pysc2.lib import actions, features
from models.base.abmodel import LearningModel
from .base.abagent import Agent
from models.base.abmodel import LearningModel
from agents.rewards.abreward import RewardBuilder

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import error


class SC2Agent(Agent):
    def __init__(self, model: LearningModel, reward_builder: RewardBuilder, observation_spec, action_spec):
        super(SC2Agent, self).__init__(model, reward_builder)
        self.obs_spec = observation_spec
        self.action_spec = action_spec
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

    def reset(self):
        super(SC2Agent, self).reset()
        self.episodes += 1

    def step(self, obs, reward, done):
        self.steps += 1
        self.reward += reward

        if self.action_wrapper.is_action_done():
            ## Building our agent's state
            current_state = self.build_state(obs)

            excluded_actions = self.action_wrapper.get_excluded_actions(obs)
            predicted_action_idx = self.model.choose_action(current_state, excluded_actions)

            self.previous_action = predicted_action_idx
            self.previous_state = current_state

        selected_action = [self.action_wrapper.get_action(self.previous_action, obs)]

        try:
            action_id = selected_action[0].function
        except:
            raise error.ActionError("Invalid function structure. Function name: %s." % selected_action[0])
        return selected_action

    def play(self, obs):
        if self.action_wrapper.is_action_done():
            current_state = self.build_state(obs)
            excluded_actions = self.action_wrapper.get_excluded_actions(obs)
            predicted_action_idx = self.model.predict(current_state, excluded_actions)
            self.previous_action = predicted_action_idx
        arrayed_action = [self.action_wrapper.get_action(self.previous_action, obs)]
        return arrayed_action

