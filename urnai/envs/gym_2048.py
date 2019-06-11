'''
Custom Env for the classic 2048 game made by Sanyam Kapoor, integrated with OpenAi's, avaliable at https://pypi.org/project/gym-2048/
This class is an exact copy of the GymEnv class with exception to the top level import
of the gym_2048 library, so that we can separate the usage of original gym games from
aditional games added by the community.
'''

import gym_2048
from .gym import GymEnv

class GymEnv2048(GymEnv):
    def __init__(self, _id, render=False, reset_done=True, num_steps=100):
        super().__init__(_id, render, reset_done)
        self.num_steps = num_steps

        self.start()
