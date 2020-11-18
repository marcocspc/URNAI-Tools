from .abneuralnetwork import ABNeuralNetwork 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PyTorchDeepNeuralNetwork(ABNeuralNetwork):

    def __init__(self):
        super().__init__()
        #these lines are needed to setup
        #pytorch
        self.aux_pytorch_obj = SubDeepQNetwork()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_layers = []

    def add_input_layer(self, idx):
        self.aux_pytorch_obj.model_layers.append(nn.Linear(self.state_size, self.build_model[idx]['nodes']))

    def add_output_layer(self, idx):
        self.aux_pytorch_obj.model_layers.append(nn.Linear(self.build_model[idx-1]['nodes'], self.action_space_size))

    def add_fully_connected_layer(self, idx):
        self.aux_pytorch_obj.model_layers.append(nn.Linear(self.build_model[idx-1]['nodes'], self.build_model[idx]['nodes']))

    #TODO
    #def add_convolutional_layer(self, idx):

    #TODO
    #def update(self, s, a, r, s_) -> None :

    def get_output(self, state) -> list:
        x = state
        for i in range(len(self.model_layers) - 1):
            x = F.relu(self.model_layers[i](x))
        return self.model_layers[-1](x)

    #TODO
    #def set_seed(self) -> None:

    class SubDeepQNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_layers = nn.ModuleList()
