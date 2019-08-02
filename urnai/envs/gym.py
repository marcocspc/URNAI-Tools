import gym
from .base.abenv import Env

class GymEnv(Env):
    def __init__(self, id, render=False, reset_done=True, num_steps=100):
        super().__init__(id, render, reset_done)
        self.num_steps = num_steps

        self.start()
        
    
    def start(self):
        if not self.env_instance:
            self.env_instance = gym.make(self.id)

    
    def step(self, action):
        obs, reward, done, _ = self.env_instance.step(action)
        return obs, reward, done


    def reset(self):
        return self.env_instance.reset()

    
    def close(self):
        self.env_instance.close()

    
    def restart(self):
        self.close()
        self.reset()

