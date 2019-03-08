import tensorflow as tf
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import features
from envs import sc2
from agents import terran_attack_agent_sparse as terran
from agents.actions.sc2_wrapper import SC2Wrapper
from models.ql_table_model import QLearningTable
from agents.terran_attack_agent_sparse import TerranAgentSparse
from agents.terran_attack_agent import TerranAgent
from models.dql_model import DQNetwork


def main(unused_argv):

    action_wrapper = SC2Wrapper()
    #agent = TerranAgentSparse(action_wrapper)
    agent = TerranAgent(action_wrapper)
    dq_network = DQNetwork(agent, './saved/terran_dql_model')
    #q_table = QLearningTable(agent, './saved/terran_attack_model')

    players = [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)]

    try:
        while True:
            env = sc2.SC2Env(map_name="Simple64", players=players, render=True)
            env.start()

            with env.env_instance as env_instance:
                agent.setup(env_instance.observation_spec(), env_instance.action_spec())

                timesteps = env.reset()

                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]

                    if timesteps[0].last():
                        break

                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
