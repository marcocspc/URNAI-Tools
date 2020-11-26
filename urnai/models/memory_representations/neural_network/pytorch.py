from .abneuralnetwork import ABNeuralNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PyTorchDeepNeuralNetwork(ABNeuralNetwork):

    def __init__(self, action_output_size, state_input_shape, build_model, gamma, alpha, seed = None):
        #Initializing self.aux_pytorch_obj so that it can be referenced by the methods of our base class (ABNeuralNetwork)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aux_pytorch_obj = self.SubDeepQNetwork()
        
        super().__init__(action_output_size, state_input_shape, build_model, gamma, alpha)
        
        #these lines are needed to setup pytorch
        
        self.optimizer = optim.Adam(self.aux_pytorch_obj.model_layers.parameters(), lr=self.alpha)

    def add_input_layer(self, idx):
        self.aux_pytorch_obj.model_layers.append(nn.Linear(self.state_input_shape, self.build_model[idx]['nodes']).to(self.device))

    def add_output_layer(self, idx):
        self.aux_pytorch_obj.model_layers.append(nn.Linear(self.build_model[idx-1]['nodes'], self.action_output_size).to(self.device))

    def add_fully_connected_layer(self, idx):
        self.aux_pytorch_obj.model_layers.append(nn.Linear(self.build_model[idx-1]['nodes'], self.build_model[idx]['nodes']).to(self.device))

    #TODO
    #def add_convolutional_layer(self, idx):

    def update(self, state, target_output):
        # transform our state from numpy array to pytorch tensor and then feed it to our model (aux_pytorch_obj)
        # the result of this is our expected output
        torch_state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        expected_output = self.aux_pytorch_obj(torch_state)

        # transform our target output from numpy array to pytorch tensor
        target_output = torch.from_numpy(target_output).float().unsqueeze(0).to(self.device)

        # calculate loss using expected_output and target_output
        loss = torch.nn.MSELoss()(expected_output, target_output).to(self.device)

        # using loss to update the neural network (doing an optmizer step)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_output(self, state):
        # convert numpy format to something that pytorch understands
        # TODO:This line takes a long time to process when in debug mode, so it probably uses lots of processing power. Try to find an alternative to it.
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # put the network into evaluation mode
        self.aux_pytorch_obj.eval()
        # get network output
        with torch.no_grad():
            action_values = self.aux_pytorch_obj(state)
        # put the network back to training mode again
        self.aux_pytorch_obj.train()
        # return the output
        return action_values.cpu().data.numpy()

    #TODO
    def set_seed(self, seed):
        pass

    class SubDeepQNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_layers = nn.ModuleList()

        def forward(self,x):
            for i in range(len(self.model_layers) - 1):
                test = self.model_layers[i]
                x = F.relu(self.model_layers[i](x))
            return self.model_layers[-1](x)
