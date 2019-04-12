from absl import app
from pysc2.env import sc2_env
from pysc2.lib import features
from envs.sc2 import SC2Env
from envs.gym import GymEnv
from envs.trainer import Trainer
from agents import terran_attack_agent_sparse as terran
from agents.actions.sc2_wrapper import SC2Wrapper
from agents.actions.gym_wrapper import GymWrapper
from models.ql_table_model import QLearningTable
from agents.cartpole_agent import CartpoleAgent
from agents.frozenlake_agent import FrozenLakeAgent
from agents.terran_attack_agent_sparse import TerranAgentSparse
from agents.terran_attack_agent import TerranAgent
from models.dql_model import DQNetwork
from models.dql_working import DQNWorking
from models.dql_keras import DQNKeras

def main(unused_argv):
    trainer = Trainer()

    # try:
    #     # Initializing our FrozenLake enviroment
    #     env = GymEnv(_id="CartPole-v1")

    #     # Initializing our FrozenLake agent
    #     wrapper = GymWrapper(env)
    #     agent = CartpoleAgent(wrapper, env)

    #     # Initializing a Deep Q-Learning model using our agent
    #     dq_network = DQNKeras(agent, 'src/models/saved/cartpole_dql_working', gamma=0.95, epsilon_decay=0.995, epsilon_min=0.1)

    #     # Using Trainer to train and play with our agent.
    #     trainer.train(env, agent, num_episodes=4500, max_steps=1000, save_steps=1000)
    #     trainer.play(env, agent, num_matches=100, max_steps=1000)
    # except KeyboardInterrupt:
    #     pass

    try:
        # Initializing our FrozenLake enviroment
        env = GymEnv(_id="FrozenLakeNotSlippery-v0")

        # Initializing our FrozenLake agent
        frozen_wrapper = GymWrapper(env)
        agent = FrozenLakeAgent(frozen_wrapper)

        # Initializing a Deep Q-Learning model using our agent
        dq_network = DQNWorking(agent, 'src/models/saved/frozenlake_dql_working')

        # Using Trainer to train and play with our agent.
        trainer.train(env, agent, num_episodes=10000, max_steps=100, save_steps=1000)
        trainer.play(env, agent, num_matches=100, max_steps=100)
    except KeyboardInterrupt:
        pass

    # try:
    #     ## Initializing our StarCraft 2 environment
    #     players = [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)]
    #     env = SC2Env(map_name="Simple64", players=players, render=True)
    #     action_wrapper = SC2Wrapper()

    #     ## Terran sparse agent with a Deep Q-Learning model
    #     agent = TerranAgentSparse(action_wrapper)
    #     dq_network = DQNetwork(agent, 'src/models/saved/terran_sparse_dql')

    #     trainer.train(env, agent, num_episodes=10, save_steps=1)
    #     trainer.play(env, agent, num_matches=2)
    # except KeyboardInterrupt:
    #     pass


    # try:
    #     ## Initializing our StarCraft 2 environment
    #     players = [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)]
    #     env = SC2Env(map_name="Simple64", players=players, render=True)
    #     action_wrapper = SC2Wrapper()

    #     ## Terran agent with a Deep Q-Learning model
    #     agent = TerranAgent(action_wrapper)
    #     dq_network = DQNetwork(agent, 'src/models/saved/terran_dql')

    #     trainer.train(env, agent, num_episodes=10, max_steps=None, save_steps=1)
    #     trainer.play(env, agent, num_matches=2, max_steps=None)
    # except KeyboardInterrupt:
    #     pass



if __name__ == '__main__':
    app.run(main)
