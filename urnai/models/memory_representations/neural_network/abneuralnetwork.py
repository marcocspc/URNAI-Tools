from abc import abstractmethod

from models.memory_representations.base.abmem import ABMemoryRepresentation
from urnai.utils.error import IncoherentBuildModelError
from urnai.utils.error import UnsupportedBuildModelLayerTypeError
from models.model_builder import ModelBuilder

class ABNeuralNetwork(ABMemoryRepresentation):

    def make_model(self):
        if self.build_model[0]['type'] == ModelBuilder.LAYER_INPUT and self.build_model[-1]['type'] == ModelBuilder.LAYER_OUTPUT:
            self.build_model[0]['shape'] = [None, self.state_input_shape]
        else:
            raise IncoherentBuildModelError("The first layer type should be {} and the last one type should be {}".format(ModelBuilder.LAYER_INPUT, ModelBuilder.LAYER_OUTPUT))
        
        self.build_model[-1]['length'] = self.action_output_size

        for idx, (layer_model) in enumerate(self.build_model):
            if layer_model['type'] == ModelBuilder.LAYER_INPUT: 
                if idx == 0:
                    self.add_input_layer(idx)
                else:
                    raise IncoherentBuildModelError("The first layer type should be {}".format(ModelBuilder.LAYER_INPUT))
            elif layer_model['type'] == ModelBuilder.LAYER_OUTPUT:
                if (idx + 1) == len(self.build_model):
                    self.add_output_layer(idx)
                else:
                    raise IncoherentBuildModelError("The last layer type should be {}".format(ModelBuilder.LAYER_OUTPUT))
            elif layer_model['type'] == ModelBuilder.LAYER_FULLY_CONNECTED:
                self.add_fully_connected_layer(idx)
            elif layer_model['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                self.add_convolutional_layer(idx)

    @abstractmethod
    def add_input_layer(self, idx) -> None:
        ...

    @abstractmethod
    def add_output_layer(self, idx) -> None:
        ...

    def add_fully_connected_layer(self, idx):
        raise UnsupportedBuildModelLayerTypeError("Fully-connected layer is not supported by {}.".format(self.__class__.__name__))

    def add_convolutional_layer(self, idx):
        raise UnsupportedBuildModelLayerTypeError("Convolutional layer is not supported by {}.".format(self.__class__.__name__))
