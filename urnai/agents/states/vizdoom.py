import numpy as np
from .abstate import StateBuilder
from envs.base.abenv import Env

class DefaultVizDoomState(StateBuilder):
    def build_state(self, obs):
        return obs

    def get_state_dim(self):
        return len(obs)

class TFVizDoomHealthGatheringState(StateBuilder):
    def build_state(self, obs):
        return obs.game_variables.reshape(1, self.get_state_dim())

    def get_state_dim(self):
        return 17

class TFVizDoom2CustomState(StateBuilder):
    def build_state(self, obs):
        return obs.game_variables.reshape(1, self.get_state_dim())

    def get_state_dim(self):
        return 17
