import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
from .base.abenv import Env

class FrozenLakeEnv(Env):
    def __init__(self, _id, render=False, reset_done=True, num_episodes=None, num_steps=100):
        super().__init__(_id, render, reset_done, num_episodes)
        self.env_instance = None
        self.num_steps = num_steps

        self.start()
        print(self.env_instance.action_space)
        print(self.env_instance.action_space.n)


    
    def start(self):
        register(
            id='FrozenLakeNotSlippery-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name' : '4x4', 'is_slippery': False},
            max_episode_steps=self.num_steps,
            reward_threshold=0.78, # optimum = .8196
        )

        self.env_instance = gym.make("FrozenLakeNotSlippery-v0")

    
    def step(self, action):
        '''
        Executes an action on the environment and returns the observation.
        '''
        return self.env_instance.step(action)


    def reset(self):
        return self.env_instance.reset()

    
    def stop(self):
        self.env_instance.close()

    
    def restart(self):
        self.stop()
        self.reset()

    
    def train(self, agent, save_steps=1000):
        num_steps = self.num_steps

        # List of rewards
        rewards = []

        #lista das medias de recompensa para imprimir o grafico
        reward_mean = [0]

        #lista do numero de vitorias
        victories = [0]
        victory_percentage = [0]

        print("Treinando...")

        try:
        # 2 For life or until learning is stopped
            episode = 0
            while episode < self.num_episodes:

                print("Episódio " + str(episode + 1) + " de " + str(self.num_episodes), end = "\r")

                # Reset the environment
                obs = self.reset()
                reward = 0
                done = False


                total_rewards = 0
                victory = False
                
                
                for step in range(num_steps):
                    
                    if step == num_steps - 1:
                        done = True

                    action = agent.step(obs, reward, done)

                    # If done (if we're dead) : finish episode
                    if done:
                        victory = reward == 1
                        break
                    else:
                        # Take the action (a) and observe the outcome state(s') and reward (r)
                        obs, reward, done, _ = self.step(action)
                        total_rewards += reward
                episode += 1

                if episode % save_steps == 0:
                    agent.model.save()

                rewards.append(total_rewards)
                reward_mean.append(sum(rewards)/(episode + 1))

                if victory:
                    victories.append(1)
                else:
                    victories.append(0)
                victory_percentage.append(sum(victories)/(episode + 1))


        except KeyboardInterrupt:
            print()

        print()
        print("Treino finalizado!")
        print("Recompensa acumulada: " + str(sum(rewards) / self.num_episodes))
        print("Taxa de vitórias: " + str(victory_percentage[-1]))
        print()


    def play(self, agent, num_matches):
        max_steps = self.num_steps

        rewards = []
        reward_mean = [0]

        victories = [0]
        victory_percentage = [0]

        print("Jogando...")

        try:
        # 2 For life or until learning is stopped
            for match in range(num_matches):
                print("Partida " + str(match + 1) + " de " + str(num_matches), end = "\r")

                # Reset the environment
                obs = self.reset()
                reward = 0
                done = False

                total_rewards = 0
                victory = False
                
                
                for step in range(max_steps):

                    action = agent.play(obs, reward, done)

                    # If done (if we're dead) : finish episode
                    if done:
                        victory = reward == 1
                        break
                    else:
                        # Take the action (a) and observe the outcome state(s') and reward (r)
                        obs, reward, done, _ = self.step(action)
                        total_rewards += reward


                rewards.append(total_rewards)
                reward_mean.append(sum(rewards)/(match + 1))

                if victory:
                    victories.append(1)
                else:
                    victories.append(0)
                victory_percentage.append(sum(victories)/(match + 1))

        except KeyboardInterrupt:
            print()

        print()
        print("Partidas finalizadas!")
        print("Recompensa acumulada: " + str(sum(rewards)/num_matches))
        print("Taxa de vitórias: " + str(victory_percentage[-1]))
        print()

