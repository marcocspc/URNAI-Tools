import numpy as np
import random
import os
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
# from keras.optimizers import Adam
from .base.abmodel import LearningModel
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder
from .model_builder import ModelBuilder
from urnai.utils.error import IncoherentBuildModelError
from urnai.utils.error import UnsupportedBuildModelLayerTypeError


class DQNPytorch(LearningModel):
    """
    A Deep Q-Network implemented made using PyTorch.
    This implementation was based on Unnat Singh's Deep Q-Network implementation (https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda)

    Attributes
    ----------
    action_wrapper : ActionWrapper
        an action wrapper
    state_builder : StateBuilder
        a state builder
    learning_rate : float
        the learning rate
    gamma : float
        the discount factor
    name : str
        class name
    epsilon_start : float
        maximum value of our epsilon greedy strategy (1.0 = 100% random actions)
    epsilon_min : float
        minimum value of our epsilon greedy strategy (0.01 = 1% random actions)
    epsilon_decay : float
        value that our espilon greedy will decay by. It will be multiplied by our current epsilon to determine the next one (0.995 = 0.5% decay)
    per_episode_epsilon_decay : bool
        determines if the epsilon decay will happen at every game step (false) or only at the end of the episode (true)
    batch_size : int
        size of a memory sample batch that will be taken every game step to train the model
    memory_maxlen : int
        maximum size of our memory
    min_memory_size : int
        minimum size of memory before the model starts to learn
    build_model : ModelBuilder
        a dictionary describing the models structure (layers, nodes etc)

    Methods
    -------
    learn(s, a, r, s_, done)
        goes trough the learning process of the DQL algorithm using PyTorche's functionalities
    """

    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, learning_rate=0.001, gamma=0.99, 
                name='DQNPytorch', epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995, per_episode_epsilon_decay=False, batch_size=64,
                memory_maxlen=50000, min_memory_size=1000, build_model = ModelBuilder.DEFAULT_BUILD_MODEL):
        super(DQNPytorch, self).__init__(action_wrapper, state_builder, gamma, learning_rate, epsilon_start, epsilon_min, epsilon_decay, per_episode_epsilon_decay, name)

        self.build_model = build_model
        self.model = self.make_model()
        # Target model, used in .predict every step
        self.target_model = self.make_model()
        
        self.memory = deque(maxlen=memory_maxlen)
        self.memory_maxlen = memory_maxlen
        self.min_memory_size = min_memory_size
        self.batch_size = batch_size

        self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def make_model(self):
        # @classmethod
        # def new_init(self, model_layers):
        #     super(self.__class__, self).__init__()
        #     for i in range(eval("self.number_layers")):
        #         setattr(self, "layer"+str(i), model_layers[i])

        # @classmethod
        # def forward(self, x):
        #     for i in range(eval("self.number_layers - 1")):
        #         x = F.relu(eval("self.layer"+str(i)+"(x)"))
        #     return eval("self.layer"+str(i+1)+"(x)")

        # model_layers = []

        # if self.build_model[0]['type'] == ModelBuilder.LAYER_INPUT and self.build_model[-1]['type'] == ModelBuilder.LAYER_OUTPUT:
        #     self.build_model[0]['shape'] = [None, self.state_size]
        #     self.build_model[-1]['length'] = self.action_size

        # for idx, (layer_model) in enumerate(self.build_model):
        #     if layer_model['type'] == ModelBuilder.LAYER_INPUT: 
        #         if self.build_model.index(layer_model) == 0:
        #             model_layers = [nn.Linear(self.state_size, layer_model['nodes'])]
        #         else:
        #             raise IncoherentBuildModelError("Input Layer must be the first one.") 
        #     elif layer_model['type'] == ModelBuilder.LAYER_FULLY_CONNECTED:
        #         model_layers.append(nn.Linear(self.build_model[idx]['nodes'], layer_model['nodes']))

        #     elif layer_model['type'] == ModelBuilder.LAYER_OUTPUT:
        #         model_layers.append(nn.Linear(self.build_model[idx]['length'], self.action_size))
        #     elif layer_model['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
        #         if self.build_model.index(layer_model) == 0:
        #             model_layers.append(nn.Sequential(
        #                 nn.Conv2d(in_channels=layer_model['input_shape'][2], out_channels=layer_model['filters'], kernel_size=layer_model['filter_shape'], stride=1, padding=2),
        #                 nn.ReLU(),
        #                 nn.MaxPool2d(kernel_size=layer_model['max_pooling_pool_size_shape'], stride=2)
        #             ))
        #         else:
        #             model_layers.append(nn.Sequential(
        #                 nn.Conv2d(in_channels=self.build_model[idx]['filters'], out_channels=layer_model['filters'], kernel_size=layer_model['filter_shape'], stride=1, padding=2),
        #                 nn.ReLU(),
        #                 nn.MaxPool2d(kernel_size=layer_model['max_pooling_pool_size_shape'], stride=2)
        #             ))
        #             if ModelBuilder.is_last_conv_layer(layer_model, self.build_model):
        #                 model_layers.append(nn.Dropout())
        #     else:
        #         raise UnsupportedBuildModelLayerTypeError("Unsuported Layer Type " + layer_model['type'])

        # attributes = {
        #     'number_layers': len(model_layers),
        #     'forward': forward,
        #     '__init__': new_init
        #     }
        # # for idx, layer in enumerate(model_layers):
        # #     attributes['layer'+ str(idx)] = layer

        # DQNClassPytorch = type('inheritnnModule', (nn.Module,), attributes)
        # model = DQNClassPytorch(model_layers).to(device)

        model = QNetwork(self.state_size, self.action_size).to(device)

        return model

    def memorize(self, state, action, reward, next_state, done):
        experience = self.experiences(state, action, reward, next_state, done)
        self.memory.append(experience)

    def learn(self, s, a, r, s_, done):
        self.memorize(s, a, r, s_, done)
        if len(self.memory) < self.min_memory_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in minibatch if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in minibatch if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in minibatch if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in minibatch if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in minibatch if e is not None]).astype(np.uint8)).float().to(device)

        criterion = torch.nn.MSELoss()
        self.model.train()
        self.target_model.eval()

        predicted_targets = self.model(states).gather(1,actions)

        with torch.no_grad():
            labels_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)

        labels = rewards + (self.gamma * labels_next*(1-dones))

        loss = criterion(predicted_targets, labels).to(device)
        optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #to do: add tau to model definition so that it can be passed here
        self.soft_update(self.model, self.target_model)

        if not self.per_episode_epsilon_decay:
            self.decay_epsilon()

    def soft_update(self, local_model, target_model, tau=1e-3):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

    def choose_action(self, state, excluded_actions=[]):
        if np.random.rand() <= self.epsilon_greedy:
            random_action = random.choice(self.actions)
            # Removing excluded actions
            while random_action in excluded_actions:
                random_action = random.choice(self.actions)
            return random_action
        else:
            return self.predict(state, excluded_actions)
        
    def predict(self, state, excluded_actions=[]):
        '''
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()

        return np.argmax(action_values.cpu().data.numpy())
        
    def save_extra(self, persist_path):
        torch.save(self.model.state_dict(), self.get_full_persistance_path(persist_path))

    def load_extra(self, persist_path):
        exists = os.path.isfile(self.get_full_persistance_path(persist_path))

        if(exists):
            self.model = self.make_model()
            self.target_model = self.make_model()
            self.model.load_state_dict(torch.load(self.get_full_persistance_path(persist_path)))
            self.target_model.load_state_dict(torch.load(self.get_full_persistance_path(persist_path)))

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_unit=64, fc2_unit = 64):
        super(QNetwork,self).__init__()
        self.fc1= nn.Linear(state_size,fc1_unit)
        self.fc2 = nn.Linear(fc1_unit,fc2_unit)
        self.fc3 = nn.Linear(fc2_unit,action_size)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)