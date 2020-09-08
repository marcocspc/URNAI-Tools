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
from agents.actions.sc2_wrapper import SC2Wrapper, TerranWrapper, SimpleTerranWrapper, ProtossWrapper
from agents.rewards.sc2 import KilledUnitsReward, KilledUnitsRewardBoosted, GeneralReward
from agents.states.sc2 import Simple64State
from agents.states.sc2 import Simple64StateFullRes
from agents.states.sc2 import Simple64GridState
from models.pg_keras import PGKeras
from models.dqn_keras import DQNKeras
from models.ddqn_keras import DDQNKeras
from utils.functions import query_yes_no
from models.model_builder import ModelBuilder

from urnai.tdd.reporter import Reporter as rp

""" Change "sc2_local_path" to your local SC2 installation path. 
If you used the default installation path, you may ignore this step.
For more information consult https://github.com/deepmind/pysc2#get-starcraft-ii 
"""
sc2_local_path = "D:/Program Files (x86)/StarCraft II"
rp.VERBOSITY_LEVEL = 0

def main(unused_argv):
    try:
        ## Checking whether or not to change SC2's instalation path environment variable
        ## This only needs to be done once on each machine
        # if query_yes_no("Change SC2PATH to " + sc2_local_path + " ?"):
        #     os.environ["SC2PATH"] = sc2_local_path

        ## Initializing our StarCraft 2 environment
        env = SC2Env(map_name="Simple64", render=False, step_mul=16, player_race="terran", enemy_race="random", difficulty="very_easy")
        
        action_wrapper = SimpleTerranWrapper()
        state_builder = Simple64GridState(grid_size=4)
        
        helper = ModelBuilder()
        helper.add_input_layer(int(state_builder.get_state_dim()), nodes=50)
        helper.add_fullyconn_layer(nodes=50)
        helper.add_output_layer(action_wrapper.get_action_space_dim())

        dq_network = DDQNKeras(action_wrapper=action_wrapper, state_builder=state_builder, build_model=helper.get_model_layout(), per_episode_epsilon_decay=False,
                            gamma=0.99, learning_rate=0.001, epsilon_decay=0.99999, epsilon_min=0.005, memory_maxlen=100000, min_memory_size=2000)
        
        # Terran agent
        agent = SC2Agent(dq_network, KilledUnitsReward())

        #trainer = Trainer(env, agent, save_path='/home/lpdcalves/', file_name="terran_ddqn_v_easy", save_every=100, enable_save=True)
        trainer = Trainer(env, agent, save_path='urnai/models/saved', file_name="terran_ddqn_test_persist_path3", save_every=20, enable_save=True, relative_path=True)
        trainer.train(num_episodes=3000, max_steps=1200)
        trainer.play(num_matches=100, max_steps=1200)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
