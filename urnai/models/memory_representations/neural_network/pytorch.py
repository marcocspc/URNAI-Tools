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
        self.optimizer = optim.Adam(self.aux_pytorch_obj.model_layers.parameters(), lr=self.alpha)

    def add_input_layer(self, idx):
        self.aux_pytorch_obj.model_layers.append(nn.Linear(self.state_size, self.build_model[idx]['nodes']))

    def add_output_layer(self, idx):
        self.aux_pytorch_obj.model_layers.append(nn.Linear(self.build_model[idx-1]['nodes'], self.action_space_size))

    def add_fully_connected_layer(self, idx):
        self.aux_pytorch_obj.model_layers.append(nn.Linear(self.build_model[idx-1]['nodes'], self.build_model[idx]['nodes']))

    #TODO
    #def add_convolutional_layer(self, idx):

    def update(self, state, target_output):
        # get model expected output
        expected_output = self.get_output(state) 

        # calculate loss using expected_output and target_output
        loss = torch.nn.MSELoss()(expected_output, target_output).to(self.device)

        # using loss to update the neural network (doing an optmizer step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def get_output(self, state):
        # convert numpy format to something that pytorch understands
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
    #def set_seed(self) -> None:

    class SubDeepQNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_layers = nn.ModuleList()

        def forward(self,x):
            for i in range(len(self.model_layers) - 1):
                x = F.relu(self.model_layers[i](x))
            return self.model_layers[-1](x)
