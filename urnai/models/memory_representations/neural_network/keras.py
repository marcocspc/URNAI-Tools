from .abneuralnetwork import ABNeuralNetwork 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from keras.optimizers import Adam
from keras import backend as K

class KerasDeepNeuralNetwork(ABNeuralNetwork):
    """
    Implementation of a Deep Neural Network using Keras

    Parameters:
        - action_output_size: Number of output actions
        - state_input_shape: shape of the input
        - build_model: A dict representing the NN's layers. Can be generated by the ModelBuilder.get_model_layout() method from an instantiated ModelBuilder object.
        - gamma: gamma parameter in the Deep Q Learning algorithm
        - alpha: learning rate
        - seed: (default = none)
    """

    def __init__(self, action_output_size, state_input_shape, build_model, gamma, alpha, seed = None):     
        super().__init__(action_output_size, state_input_shape, build_model, gamma, alpha)

        self.model.compile(optimizer=Adam(lr=self.alpha), loss='mse', metrics=['accuracy'])


    def add_input_layer(self, idx):
        self.model.add(Dense(self.build_model[idx]['nodes'], input_dim=self.build_model[idx]['shape'][1], activation='relu'))

    def add_output_layer(self, idx):
        self.model.add(Dense(self.build_model[idx]['length'], activation='linear'))

    def add_fully_connected_layer(self, idx):
        self.model.add(Dense(self.build_model[idx-1]['nodes'], activation='relu'))

    #TODO
    #def add_convolutional_layer(self, idx):

    #TODO
    def update(self, state, target_output):
        # get model expected output
        expected_output = self.get_output(state) 

        # calculate loss using expected_output and target_output
        """loss = torch.nn.MSELoss()(expected_output, target_output).to(self.device)"""

        # using loss to update the neural network (doing an optmizer step)
        """optimizer.zero_grad()
        loss.backward()
        optimizer.step()"""


    def get_output(self, state):
        return self.model.predict(state)

    #TODO
    def set_seed(self, seed) -> None:
        pass

    def create_base_model(self):
        model = Sequential()
        return model

class AA(KerasDeepNeuralNetwork):
    def __init__(self, action_output_size, state_input_shape, build_model, gamma, alpha, seed = None):        
        super().__init__(action_output_size, state_input_shape, build_model, gamma, alpha)
    
    def make_model(self):
        self.model = Sequential()

        self.model.add(Dense(50, activation='relu', input_dim=self.state_input_shape))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(self.action_output_size, activation='linear'))

        self.model.compile(optimizer=Adam(lr=self.alpha), loss='mse', metrics=['accuracy'])