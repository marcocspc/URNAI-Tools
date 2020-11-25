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

    #TODO
    def update(self, state, expected_output):
        #obter saída da rede
        predicted_output = self.get_output(state) 

        #usar saída da rede e saída esperada para calcular o loss
        loss = torch.nn.MSELoss()(predicted_output, expected_output).to(self.device)

        #usar o loss e o learning_rate (alpha) pra atualizar a rede 
        optimizer = optim.Adam(self.aux_pytorch_obj.parameters(),lr=self.alpha)

        #deus queira que as linhas abaixo estejam atualizando a rede
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    def get_output(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.aux_pytorch_obj.eval()
        with torch.no_grad():
            action_values = self.aux_pytorch_obj(state)
        self.aux_pytorch_obj.train()

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
