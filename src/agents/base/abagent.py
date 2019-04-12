from abc import ABC, abstractmethod
from agents.actions.base.abwrapper import ActionWrapper

class Agent(ABC):
    
    def __init__(self, action_wrapper: ActionWrapper):
        self.model = None
        self.previous_action = None
        self.previous_state = None
        self.action_wrapper = action_wrapper

    @abstractmethod
    def build_state(self, obs):
        pass

    @abstractmethod
    def get_reward(self, obs, reward, done):
        pass

    @abstractmethod
    def get_state_dim(self):
        pass

    def setup(self, env):
        '''
        All agents need to have a setup method because PySC2 agents require a setup method, so this is just boilerplate code
        required for the Trainer class to work, since the setup method must be called inside both play and train methods for PySC2 agents to work.
        '''
        i = 0


    def reset(self):
        self.previous_action = None
        self.previous_state = None


    def learn(self, obs, reward, done):
        if self.previous_state is not None:
            next_state = self.build_state(obs)
            reward = self.get_reward(obs, reward, done)
            self.model.learn(self.previous_state, self.previous_action, reward, next_state, done)


    def play(self, obs):
        if self.action_wrapper.is_action_done():
            current_state = self.build_state(obs)
            predicted_action = self.model.predict(current_state)
            self.previous_action = predicted_action
        return self.action_wrapper.get_action(predicted_action, obs)


    def step(self, obs, obs_reward, done):
        # Taking the first step for a smart action
        if self.action_wrapper.is_action_done():
            ## Building our agent's current state
            current_state = self.build_state(obs)
            
            # If it's not the first step, we can learn
            excluded_actions = self.action_wrapper.get_excluded_actions(obs)
            current_action = self.model.choose_action(current_state, excluded_actions)

            self.previous_action = current_action
            self.previous_state = current_state
        return self.action_wrapper.get_action(self.previous_action, obs)


    def set_model(self, model):
        self.model = model
