from .abneuralnetwork import ABNeuralNetwork 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from keras.optimizers import Adam
from keras import backend as K

class AA(KerasDeepNeuralNetwork):
    def __init__(self, action_output_size, state_input_shape, build_model, gamma, alpha, seed = None):        
        super().__init__(action_output_size, state_input_shape, build_model, gamma, alpha)
    
    def make_model(self):
        self.model = Sequential()

        self.model.add(Dense(50, activation='relu', input_dim=self.state_input_shape))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(self.action_output_size, activation='linear'))

        self.model.compile(optimizer=Adam(lr=self.alpha), loss='mse', metrics=['accuracy'])


class KerasDeepNeuralNetwork(ABNeuralNetwork):

    def __init__(self, action_output_size, state_input_shape, build_model, gamma, alpha, seed = None):     
        super().__init__(action_output_size, state_input_shape, build_model, gamma, alpha)

        self.model.compile(optimizer=Adam(lr=self.alpha), loss='mse', metrics=['accuracy'])


    def add_input_layer(self, idx):
        self.model.add(Dense(self.build_model[idx]['nodes'], input_dim=self.build_model[idx]['shape'][1], activation='relu'))

    def add_output_layer(self, idx):
        self.model.add(Dense(self.build_model[idx-1]['length'], activation='linear'))

    def add_fully_connected_layer(self, idx):
        self.model.add(Dense(self.build_model[idx-1]['nodes'], activation='relu'))

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
        return self.model.predict(state)

    #TODO
    #def set_seed(self) -> None:

    def create_base_model(self):
        model = Sequential()
        return model

    class SubDeepQNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_layers = nn.ModuleList()

        def forward(self,x):
            for i in range(len(self.model_layers) - 1):
                x = F.relu(self.model_layers[i](x))
            return self.model_layers[-1](x)
