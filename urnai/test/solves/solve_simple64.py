import os,sys
sys.path.insert(0, os.getcwd())

from absl import app
from pysc2.env import sc2_env
from urnai.envs.sc2 import SC2Env
from urnai.trainers.trainer import Trainer
from urnai.agents.sc2_agent import SC2Agent
from urnai.agents.actions.sc2_wrapper import SimpleTerranWrapper
from urnai.agents.actions.mo_spatial_terran_wrapper import MOspatialTerranWrapper
from urnai.agents.rewards.sc2 import KilledUnitsReward
from urnai.agents.states.sc2 import Simple64GridState, SimpleCroppedGridState
from urnai.models.ddqn_keras import DDQNKeras
from urnai.models.ddqn_keras_mo import DDQNKerasMO
from urnai.utils.functions import query_yes_no
from urnai.models.model_builder import ModelBuilder

from urnai.tdd.reporter import Reporter as rp

""" Change "sc2_local_path" to your local SC2 installation path. 
If you used the default installation path, you may ignore this step.
For more information see https://github.com/deepmind/pysc2#get-starcraft-ii 
"""
# sc2_local_path = "D:/Program Files (x86)/StarCraft II"

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main(unused_argv):
    try:
        ## Checking whether or not to change SC2's instalation path environment variable
        ## This only needs to be done once on each machine
        # if query_yes_no("Change SC2PATH to " + sc2_local_path + " ?"):
        #     os.environ["SC2PATH"] = sc2_local_path

        ## Initializing our StarCraft 2 environment
        env = SC2Env(map_name="Simple64", render=False, step_mul=16, player_race="terran", enemy_race="random", difficulty="very_easy")
        
        action_wrapper = SimpleTerranWrapper()
        #state_builder = Simple64GridState(grid_size=4)
        state_builder = SimpleCroppedGridState(grid_size=4, x1=10, y1=10, x2=50, y2=50, r_enemy=True, r_player=True, r_neutral=False)
        
        helper = ModelBuilder()
        helper.add_input_layer(nodes=50)
        #helper.add_fullyconn_layer(nodes=50)
        helper.add_output_layer()

        dq_network = DDQNKeras(action_wrapper=action_wrapper, state_builder=state_builder, build_model=helper.get_model_layout(), per_episode_epsilon_decay=False,
                            gamma=0.99, learning_rate=0.001, epsilon_decay=0.99999, epsilon_min=0.005, memory_maxlen=100000, min_memory_size=2000)  
        
        # Terran agent
        agent = SC2Agent(dq_network, KilledUnitsReward())

        trainer = Trainer(env, agent, save_path='/home/lpdcalves/', file_name="terran_ddqn_v_easy",
                        save_every=100, enable_save=True, relative_path=False, reset_epsilon=False,
                        max_training_episodes=3000, max_steps_training=1200,
                        max_test_episodes=100, max_steps_testing=1200)

        # trainer = Trainer(env, agent, save_path='urnai/models/saved', file_name="terran_ddqn_exemplo",
        #                 save_every=20, enable_save=True, relative_path=True, reset_epsilon=False,
        #                 max_training_episodes=2, max_steps_training=300,
        #                 max_test_episodes=2, max_steps_testing=300)
        trainer.train()
        trainer.play()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
