import os, importlib
from abc import ABC, abstractmethod
from base.savable import Savable 
from models.model_builder import 

class ABMemoryRepresentation(Savable):

    def __init__(self, action_output_size, state_input_shape, build_model, gamma, alpha, seed = None):
        super().__init__()
        self.pickle_black_list = []
        self.seed_value = self.set_seed(seed)
        self.build_model = build_model
        self.gamma = gamma
        self.alpha = alpha

        self.action_output_size = action_output_size 
        self.state_input_shape = state_input_shape

        self.make_model()

    @abstractmethod
    def update(self, mem_input, expected_output ) -> None : 
        '''
        Update the model values
        for each action. Every memory representation
        uses different update strategies
        so parameters should be adjusted for each.
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

    @abstractmethod
    def make_model(self) -> None:
        '''
            The implementation on how
            to build the mememory representation
            should be written on this
            method.
        '''
