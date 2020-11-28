import os, importlib
from abc import abstractmethod
from base.savable import Savable 
from models.model_builder import ModelBuilder 

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

        self.model = None
        self.make_model()

    @abstractmethod
    def update(self, mem_input, target_output ) -> None : 
        '''
        Update the model values
        for each action. Every memory representation
        uses different update strategies
        so parameters should be adjusted for each.

        The mem_input atribute refers to the input that the memory representation is going to use to update itself, on a DNN this would be the state.
        The target_output refers to the desired output of the memory representation to the input, on a Q-Learning DNN this would be the target_q_values.
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
    def set_seed(self, seed) -> None:
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
