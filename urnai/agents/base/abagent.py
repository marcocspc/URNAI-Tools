from abc import ABC, abstractmethod
from models.base.abmodel import LearningModel
from agents.rewards.abreward import RewardBuilder

class Agent(ABC):
    
    def __init__(self, model: LearningModel, reward_builder: RewardBuilder):
        self.model = model
        self.action_wrapper = model.action_wrapper
        self.state_builder = model.state_builder
        self.previous_action = None
        self.previous_state = None
        self.reward_builder = reward_builder


    def build_state(self, obs):
        return self.state_builder.build_state(obs)

    def get_reward(self, obs, reward, done):
        return self.reward_builder.get_reward(obs, reward, done)

    def get_state_dim(self):
        return self.state_builder.get_state_dim()

    def reset(self):
        self.previous_action = None
        self.previous_state = None
        self.action_wrapper.reset()

    def learn(self, obs, reward, done, is_last_step: bool):
        if self.previous_state is not None:
            next_state = self.build_state(obs)
            reward = self.get_reward(obs, reward, done)
            self.model.learn(self.previous_state, self.previous_action, reward, next_state, done, is_last_step)


    '''
    This method should:
    1) Build a State using obs
    2) Use the state that was built to get an ActionIndex from the Agent's model
    3) Update self.previous_state with the current state and self.previous_action with the ActionIndex
    4) Return an Action from the Agent's ActionWrapper by using the ActionIndex from step 2
    '''
    @abstractmethod
    def step(self, obs, obs_reward, done):
        pass


    '''
    This method should:
    1) Build a State using obs
    2) Use the state that was built to get an ActionIndex from the Agent's model
    3) Return an Action from the Agent's ActionWrapper by using the ActionIndex from step 2
    '''
    @abstractmethod
    def play(self, obs):
        pass
