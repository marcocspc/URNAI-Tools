from absl import app
from pysc2.env import sc2_env
from pysc2.lib import features
from envs import sc2
from envs import frozenlake
from agents import terran_attack_agent_sparse as terran
from agents.frozenlake_agent import FrozenLakeAgent
from agents.actions.sc2_wrapper import SC2Wrapper
from agents.actions.gym_wrapper import GymWrapper
from models.ql_table_model import QLearningTable
from agents.terran_attack_agent_sparse import TerranAgentSparse
from agents.terran_attack_agent import TerranAgent
from models.dql_model import DQNetwork
from models.dql_working import DQNWorking

def main(unused_argv):

    # Initializing our FrozenLake agent
    frozen = frozenlake.FrozenLakeEnv("frozenlake", num_episodes=15000)
    frozen_wrapper = GymWrapper(frozen)
    frozen_agent = FrozenLakeAgent(frozen_wrapper)
    dq_network = DQNWorking(frozen_agent, 'src/models/saved/frozenlake_dql_working')

    # Initializing our StarCraft 2 agent
    #action_wrapper = SC2Wrapper()
    #agent = TerranAgentSparse(action_wrapper)
    #agent = TerranAgent(action_wrapper)
    #dq_network = DQNetwork(frozen_agent, 'src/models/saved/frozenlake_dql_working')
    #q_table = QLearningTable(agent, 'src/models/saved/frozenlake_dql_working')

    #players = [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)]
    #env = sc2.SC2Env(map_name="Simple64", players=players, render=True)

    try:
        frozen.train(frozen_agent)
        frozen.play(frozen_agent, 10)
        #env.train(agent)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
