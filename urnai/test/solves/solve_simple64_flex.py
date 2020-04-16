import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from absl import app
from pysc2.env import sc2_env
from envs.sc2 import SC2Env
from trainers.trainer import Trainer
from trainers.trainer import TestParams
from agents.sc2_agent import SC2Agent
from agents.actions.sc2_wrapper import SC2Wrapper, TerranWrapper, ProtossWrapper
from agents.rewards.sc2 import KilledUnitsReward, GeneralReward
from agents.states.sc2 import Simple64State_1
from agents.states.sc2 import Simple64State
from models.dql_tf_flexible import DqlTfFlexible
from models.model_builder import ModelBuilder
from utils.functions import query_yes_no

""" Change "sc2_local_path" to your local SC2 installation path. 
If you used the default installation path, you may ignore this step.
For more information consult https://github.com/deepmind/pysc2#get-starcraft-ii 
"""
sc2_local_path = "D:/Program Files (x86)/StarCraft II"

def main(unused_argv):
    try:
        ## Checking whether or not to change SC2's instalation path environment variable
        # if query_yes_no("Change SC2PATH to " + sc2_local_path + " ?"):
        #     os.environ["SC2PATH"] = sc2_local_path

        ## Initializing our StarCraft 2 environment
        players = [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)]
        env = SC2Env(map_name="Simple64", players=players, render=False, step_mul=128)
        
        action_wrapper = TerranWrapper()
        state_builder = Simple64State()

        helper = ModelBuilder()  
        helper.add_input_layer(int(state_builder.get_state_dim()))
        helper.add_fullyconn_layer(512)
        helper.add_fullyconn_layer(256)
        helper.add_output_layer(action_wrapper.get_action_space_dim())

        dq_network = DqlTfFlexible(action_wrapper=action_wrapper, state_builder=state_builder, learning_rate=0.005, gamma=0.9, save_path='urnai/models/saved/dql_flex_test/', file_name="terran_dql", build_model=helper.get_model_layout())

        ## Terran agent with a Deep Q-Learning model
        agent = SC2Agent(dq_network, GeneralReward(), env.env_instance.observation_spec(), env.env_instance.action_spec())

        #test_params = TestParams(num_matches=1, steps_per_test=25, max_steps=10000, reward_threshold=1000)
        trainer = Trainer(env, agent, save_path='urnai/models/saved/', file_name="terran_dql", save_every=1, enable_save=True)
        trainer.train(num_episodes=1, reward_from_env=True, max_steps=10000)
        trainer.play(num_matches=5)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
