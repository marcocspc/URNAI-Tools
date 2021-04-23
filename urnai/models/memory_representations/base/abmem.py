import os, importlib
from abc import abstractmethod
from base.savable import Savable 
from models.model_builder import ModelBuilder 

class ABMemoryRepresentation(Savable):

    def __init__(self, action_output_size, state_input_shape, build_model, gamma, alpha, seed = None, batch_size = 32):
        super().__init__()
        self.pickle_black_list = []
        self.seed_value = self.set_seed(seed)
        self.build_model = build_model
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size

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
            This method should set a seed for the current Machine Learning 
            library random number generator, as long as seed is not None.
            And should return the seed value so that it can be saved later through pickle
            
            In Keras or Tensorflow it would look like this:

            "if seed != None:
                import tensorflow as tf
                tf.random.set_seed(seed)
            return seed"

            In order to have a deterministic machine learning memory model we need not only the library to have a fixed seed,
            but also all other RNGs that the library uses, such as numpy, python etc to have fixed seeds. That is done through a
            separate generic method called "set_seeds" inside the LearningModel class (abmodel.py). This method will set seeds for 
            numpy and Python, as well as forcing CPU use when "cpu_only" is true to avoid non-determinism from GPU paralelism.

            more info on: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
        '''

    @abstractmethod
    def make_model(self) -> None:
        '''
            The implementation on how
            to build the mememory representation
            should be written on this
            method.
        '''
