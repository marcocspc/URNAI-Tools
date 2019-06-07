import numpy as np
from .abstate import State
from envs.base.abenv import Env

class DefaultVizDoomState(State):
    def build_state(self, obs):
        return obs

    def get_state_dim(self):
        return len(obs)

class TFVizDoomHealthGatheringState(State):
    def build_state(self, obs):
        return obs.game_variables.reshape(1, self.get_state_dim())

    def get_state_dim(self):
        return 17

class TFVizDoom2CustomState(State):
    def build_state(self, obs):
        return obs.game_variables.reshape(1, self.get_state_dim())

    def get_state_dim(self):
        return 17
