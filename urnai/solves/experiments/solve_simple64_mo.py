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
from urnai.agents.states.sc2 import Simple64GridState, Simple64GridState_SimpleTerran
from urnai.models.ddqn_keras import DDQNKeras
from urnai.models.ddqn_keras_mo import DDQNKerasMO
from urnai.utils.functions import query_yes_no
from urnai.models.model_builder import ModelBuilder

from urnai.utils.reporter import Reporter as rp

""" Change "sc2_local_path" to your local SC2 installation path. 
If you used the default installation path, you may ignore this step.
For more information consult https://github.com/deepmind/pysc2#get-starcraft-ii 
"""
# sc2_local_path = "D:/Program Files (x86)/StarCraft II"

def declare_trainer(): 
    ## Initializing our StarCraft 2 environment
    env = SC2Env(map_name="Simple64", render=False, step_mul=16, player_race="terran", enemy_race="random", difficulty="very_easy")
    
    action_wrapper = MOspatialTerranWrapper(10, 10, env.env_instance._interface_formats[0]._raw_resolution)
    state_builder = Simple64GridState_SimpleTerran(grid_size=7) # This state_builder with grid_size=7 will end upt with a total size of 110 ( (7*7)*2 + 12 )
    
    helper = ModelBuilder()
    helper.add_input_layer(nodes=80)
    #helper.add_fullyconn_layer(nodes=50)
    helper.add_output_layer()

    
    dq_network = DDQNKerasMO(action_wrapper=action_wrapper, state_builder=state_builder, build_model=helper.get_model_layout(), per_episode_epsilon_decay=False,
                        gamma=0.99, learning_rate=0.001, epsilon_decay=0.99999, epsilon_min=0.005, memory_maxlen=100000, min_memory_size=2000, batch_size=32)
    
    # Terran agent
    agent = SC2Agent(dq_network, KilledUnitsReward())

    trainer = Trainer(env, agent, save_path='/home/lpdcalves/', file_name="terran_ddqn_mo_v_easy",
                    save_every=100, enable_save=True, relative_path=False, reset_epsilon=False,
                    max_training_episodes=3000, max_steps_training=1200,
                    max_test_episodes=100, max_steps_testing=1200, log_actions=False)

    # trainer = Trainer(env, agent, save_path='urnai/models/saved', file_name="terran_ddqn_test_mo",
    #                 save_every=20, enable_save=True, relative_path=True, reset_epsilon=False,
    #                 max_training_episodes=1, max_steps_training=1200,
    #                 max_test_episodes=1, max_steps_testing=1200, log_actions=False)
    return trainer

def main(unused_argv):
    try:
        trainer = declare_trainer()
        trainer.train()
        trainer.play()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
