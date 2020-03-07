import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from absl import app
from pysc2.env import sc2_env
from envs.sc2 import SC2Env
from envs.trainer import Trainer
from envs.trainer import TestParams
from agents.sc2_agent import SC2Agent
from agents.actions.sc2_wrapper import SC2Wrapper, TerranWrapper, ProtossWrapper
from agents.rewards.sc2 import KilledUnitsReward, GeneralReward
from agents.states.sc2 import Simple64State_1
from agents.states.sc2 import Simple64State
from models.dql_tf import DQLTF
from utils.functions import query_yes_no

""" Change "sc2_local_path" to your local SC2 installation path. 
If you used the default installation path, you may ignore this step.
For more information consult https://github.com/deepmind/pysc2#get-starcraft-ii 
"""
sc2_local_path = "D:/Program Files (x86)/StarCraft II"

def main(unused_argv):
    trainer = Trainer()

    try:
        ## Checking whether or not to change SC2's instalation path environment variable
        if query_yes_no("Change SC2PATH to " + sc2_local_path + " ?"):
            os.environ["SC2PATH"] = sc2_local_path

        ## Initializing our StarCraft 2 environment
        players = [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.medium)]
        env = SC2Env(map_name="Simple64", players=players, render=True, step_mul=16)
        
        action_wrapper = TerranWrapper()
        state_builder = Simple64State()
        dq_network = DQLTF(action_wrapper=action_wrapper, state_builder=state_builder, save_path='urnai/models/saved/', file_name="terran_dql")

        ## Terran agent with a Deep Q-Learning model
        agent = SC2Agent(dq_network, GeneralReward(), env.env_instance.observation_spec(), env.env_instance.action_spec())

        test_params = TestParams(num_matches=1, steps_per_test=25, max_steps=10000, reward_threshold=1000)
        trainer.train(env, agent, num_episodes=10, save_steps=5, enable_save=True, reward_from_env=True, test_params=test_params, max_steps=10000)
        trainer.play(env, agent, num_matches=5)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
