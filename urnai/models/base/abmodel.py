from abc import ABC, abstractmethod
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder
from utils.returns import ActionIndex
from base.savable import Savable 

class LearningModel(Savable):

    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, gamma, learning_rate, learning_rate_min, learning_rate_decay, epsilon_start, epsilon_min, epsilon_decay_rate, per_episode_epsilon_decay=False, learning_rate_decay_ep_cutoff=0, name=None):
        super(LearningModel, self).__init__()
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learning_rate_min = learning_rate_min
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_ep_cutoff = learning_rate_decay_ep_cutoff
        self.name = name
        self.action_wrapper = action_wrapper
        self.state_builder = state_builder
        self.actions = action_wrapper.get_actions()
        self.action_size = action_wrapper.get_action_space_dim()
        self.state_size = state_builder.get_state_dim()

        # EXPLORATION PARAMETERS FOR EPSILON GREEDY STRATEGY
        self.epsilon_greedy = epsilon_start 
        self.epsilon_min = epsilon_min 
        self.epsilon_decay_rate = epsilon_decay_rate 
        self.per_episode_epsilon_decay = per_episode_epsilon_decay

    @abstractmethod
    def learn(self, s, a, r, s_, done, is_last_step: bool) -> None : 
        '''
        Implements the learning strategy of the specific reinforcement learning 
        algorithm implemented in the classes that inherit from LearningModel.
        '''
        ...


    @abstractmethod
    def choose_action(self, state, excluded_actions=[]) -> ActionIndex:
        '''
        Implements the exploration exploitation method for the model.
        '''
        pass


    @abstractmethod
    def predict(self, state, excluded_actions=[]) -> ActionIndex:
        '''
        Given a State, returns the index for the action with the highest Q-Value.
        '''
        pass

    def decay_epsilon(self):
        '''
        Implements the epsilon greedy strategy, effectivelly lowering the current
        epsilon greedy value by multiplying it by the epsilon_decay_rate 
        (the higher the value, the less it lowers the epsilon_decay).
        '''
        if self.epsilon_greedy > self.epsilon_min:
            self.epsilon_greedy *= self.epsilon_decay_rate

    def decay_lr(self):
        '''
        Implements a strategy for gradually lowering the learning rate of our model.
        This method works very similarly to decay_epsilon(), lowering the learning rate
        by multiplying it by the learnin_rate_decay.
        '''
        if self.learning_rate > self.learning_rate_min:
            self.learning_rate *= self.learning_rate_decay

    def ep_reset(self, episode=0):
        '''
        This method is mainly used to enact the decay_epsilon and decay_lr 
        at the end of every episode.
        '''
        if self.per_episode_epsilon_decay:
            self.decay_epsilon()

        if episode > self.learning_rate_decay_ep_cutoff:
            self.decay_lr()
