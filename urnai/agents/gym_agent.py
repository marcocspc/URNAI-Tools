from agents.base.abagent import Agent
from models.base.abmodel import LearningModel
from agents.rewards.abreward import RewardBase

class GymAgent(Agent):
    
    def __init__(self, model: LearningModel, reward_builder: RewardBase):
        super(GymAgent, self).__init__(model, reward_builder)


    def step(self, obs, obs_reward, done):
        if self.action_wrapper.is_action_done():
            ## Building our agent's current state
            current_state = self.build_state(obs)
            
            # Gets an action from the model using the current state
            excluded_actions = self.action_wrapper.get_excluded_actions(obs)
            current_action_idx = self.model.choose_action(current_state, excluded_actions)

            self.previous_action = current_action_idx
            self.previous_state = current_state
        # Returns the decoded action
        return self.action_wrapper.get_action(self.previous_action, obs)


    def play(self, obs):
        if self.action_wrapper.is_action_done():
            current_state = self.build_state(obs)
            predicted_action_idx = self.model.predict(current_state)
            self.previous_action = predicted_action_idx
        return self.action_wrapper.get_action(predicted_action_idx, obs)
