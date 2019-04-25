from absl import app
from pysc2.env import sc2_env
from envs.sc2 import SC2Env
from envs.trainer import Trainer
from agents.sc2_agent import SC2Agent
from agents.actions.sc2_wrapper import SC2Wrapper
from agents.rewards.sc2 import KilledUnitsReward
from agents.states.sc2 import Simple64State_1
from models.dql_working import DQNWorking

def main(unused_argv):
    trainer = Trainer()

    try:
        ## Initializing our StarCraft 2 environment
        players = [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)]
        env = SC2Env(map_name="Simple64", players=players, render=True)
        
        action_wrapper = SC2Wrapper()
        state_builder = Simple64State_1()
        dq_network = DQNWorking(action_wrapper, state_builder, 'urnai/models/saved/terran_dql')

        ## Terran agent with a Deep Q-Learning model
        agent = SC2Agent(dq_network, KilledUnitsReward(), env)

        trainer.train(env, agent, num_episodes=10, save_steps=1)
        trainer.play(env, agent, num_matches=2)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
