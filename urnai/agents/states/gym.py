import numpy as np
from .abstate import State
from envs.base.abenv import Env

class FrozenLakeState(State):
    def build_state(self, obs):
        if obs != None:
            index = obs
            obs = np.zeros((1, 16))
            obs[0][index] = 1
            return obs
        else:
            return None

    def get_state_dim(self):
        return 16


class PureState(State):
    def __init__(self, env: Env):
        self.state_dim = env.env_instance.observation_space.n

    def build_state(self, obs):
        return obs

    def get_state_dim(self):
        return self.state_dim

