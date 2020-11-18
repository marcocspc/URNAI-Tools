import os, importlib
from abc import ABC, abstractmethod
from base.savable import Savable 
from models.model_builder import 

class ABMemoryRepresentation(Savable):

    def __init__(self, action_space_size, state_size, build_model, gamma):
        super().__init__()
        self.pickle_black_list = []
        self.seed_value = self.set_seed()
        self.build_model = build_model
        self.gamma = gamma

        self.action_space_size = action_space_size 
        self.state_size = state_size 

        self.make_model()

    @abstractmethod
    def update(self, s, a, r, s_) -> None : 
        '''
        Update the model values
        for each action based on the tuple
        State, Action, Reward, New State
        '''
        ...

    @abstractmethod
    def get_output(self, state) -> list:
        '''
        This method should return the
        values for each action set
        in the model as a list.
        '''
        ...

    @abstractmethod
    def set_seed(self) -> None:
        '''
            This method should manually set the seed to
            generate a model when needed.
            For example, sometimes it is necessary
            to replicate a neural network instantiation
            on Tensorflow, for this, a seed is needed to
            be set inside some variables on the library.
            So, this method should do this.
        '''
