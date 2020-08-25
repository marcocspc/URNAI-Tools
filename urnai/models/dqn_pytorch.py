import numpy as np
import random
import os
from collections import deque
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


class DQNPytorch(LearningModel, nn.Module):

    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, learning_rate=0.001, gamma=0.99, 
                name='DQNPytorch', epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995, batch_size=64, batch_training=False,
                memory_maxlen=50000, per_episode_epsilon_decay=False, build_model = ModelBuilder.DEFAULT_BUILD_MODEL):
        super(DQNPytorch, self).__init__(action_wrapper, state_builder, gamma, learning_rate, epsilon_start, epsilon_min, epsilon_decay, per_episode_epsilon_decay, name)
        self.batch_size = batch_size
        self.batch_training = batch_training

        self.build_model = build_model
        self.model = self.make_model()
        
    def make_model(self):
        @classmethod
        def forward(self, x):
            for i in range(eval("self.number_layers - 1")) :
                x = F.relu(eval("self.layer"+str(i)+"(x)"))
            return eval("self.layer"+str(i+1)+"(x)")

        model_layers = []

        if self.build_model[0]['type'] == ModelBuilder.LAYER_INPUT and self.build_model[-1]['type'] == ModelBuilder.LAYER_OUTPUT:
            self.build_model[0]['shape'] = [None, self.state_size]
            self.build_model[-1]['length'] = self.action_size

        for idx, (layer_model) in enumerate(self.build_model):
            if layer_model['type'] == ModelBuilder.LAYER_INPUT: 
                if self.build_model.index(layer_model) == 0:
                    model_layers = [nn.Linear(self.state_size, layer_model['nodes'])]
                else:
                    raise IncoherentBuildModelError("Input Layer must be the first one.") 
            elif layer_model['type'] == ModelBuilder.LAYER_FULLY_CONNECTED:
                model_layers.append(nn.Linear(self.build_model[idx]['nodes'], layer_model['nodes']))

            elif layer_model['type'] == ModelBuilder.LAYER_OUTPUT:
                model_layers.append(nn.Linear(self.build_model[idx]['length'], self.action_size))
            elif layer_model['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                if self.build_model.index(layer_model) == 0:
                    model_layers.append(nn.Sequential(
                        nn.Conv2d(in_channels=layer_model['input_shape'][2], out_channels=layer_model['filters'], kernel_size=layer_model['filter_shape'], stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=layer_model['max_pooling_pool_size_shape'], stride=2)
                    ))
                else:
                    model_layers.append(nn.Sequential(
                        nn.Conv2d(in_channels=self.build_model[idx]['filters'], out_channels=layer_model['filters'], kernel_size=layer_model['filter_shape'], stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=layer_model['max_pooling_pool_size_shape'], stride=2)
                    ))
                    if ModelBuilder.is_last_conv_layer(layer_model, self.build_model):
                        model_layers.append(nn.Dropout())
            else:
                raise UnsupportedBuildModelLayerTypeError("Unsuported Layer Type " + layer_model['type'])

        attributes = {'number_layers': len(model_layers)}
        for idx, layer in enumerate(model_layers):
            attributes['layer'+ str(idx)] = layer

        model = type('inheritnnModule', (nn.Module,), attributes)
        model.forward = forward

        return model


    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        if not hasattr(self, 'batch_training') or not self.batch_training:
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
        else:
            inputs = np.zeros((len(minibatch), self.state_size))
            targets = np.zeros((len(minibatch), self.action_size))
            i=0

            for state, action, reward, next_state, done in minibatch:
                q_current_state = self.model.predict(state)[0]
                q_next_state = self.model.predict(next_state)[0]

                inputs[i] = state
                targets[i] = q_current_state

                if done:
                    targets[i,np.argmax(action)] = reward
                else:
                    targets[i,np.argmax(action)] = reward + self.gamma * np.max(q_next_state)

                i+=1

            loss = self.model.train_on_batch(inputs, targets)

        #Epsilon decay operation was here, moved it to "decay_epsilon()" and to "learn()"

    def learn(self, s, a, r, s_, done):
        self.memorize(s, a, r, s_, done)
        if len(self.memory) < self.min_memory_size:
            return

        if not self.per_episode_epsilon_decay:
            self.decay_epsilon()

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
        model.predict returns an array of arrays, containing the Q-Values for the actions. This function should return the
        corresponding action with the highest Q-Value.
        '''
        q_values = self.model.predict(state)[0]
        action_idx = int(np.argmax(q_values))

        while action_idx in excluded_actions:
            q_values = np.delete(q_values, action_idx)
            action_idx = int(np.argmax(q_values))

        return action_idx

    def save_extra(self, persist_path):
        self.model.save_weights(self.get_full_persistance_path(persist_path)+".h5")

    def load_extra(self, persist_path):
        exists = os.path.isfile(self.get_full_persistance_path(persist_path)+".h5")

        if(exists):
            self.model = self.make_model()
            self.model.load_weights(self.get_full_persistance_path(persist_path)+".h5")