import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from absl import app
from envs.gym import GymEnv
from trainers.trainer import Trainer
from trainers.trainer import TestParams
from agents.generic_agent import GenericAgent
from agents.actions.gym_wrapper import GymWrapper
from agents.rewards.gym import FrozenlakeReward
from agents.states.gym import FrozenLakeState
from models.dql_tf_flexible import DqlTfFlexible
from models.dql_keras_mem import DQNKerasMem
from models.ddqn_keras import DDQNKeras 
from models.model_builder import ModelBuilder
from models.dql_tf import DQLTF
from datetime import datetime

def main(unused_argv):
    try:
        env = GymEnv(id="FrozenLakeNotSlippery-v0")

        action_wrapper = env.get_action_wrapper() 
        state_builder = FrozenLakeState()

        training_date = str(datetime.now()).replace(" ","_").replace(":","_").replace(".","_")

        helper = ModelBuilder()
        helper.add_input_layer(int(state_builder.get_state_dim()))
        helper.add_fullyconn_layer(50)
        helper.add_output_layer(action_wrapper.get_action_space_dim())
        dq_network = DqlTfFlexible(action_wrapper=action_wrapper, state_builder=state_builder, learning_rate=0.005, gamma=0.9, build_model=helper.get_model_layout())
        agent = GenericAgent(dq_network, FrozenlakeReward())
        trainer = Trainer(env, agent, file_name=training_date + os.path.sep + "frozenlake_test_dql_flexible", save_every=1000, enable_save=True)
        # FrozenLake is solved when the agent is able to reach the end of the maze 100% of the times
        trainer.train(num_episodes=3000, reward_from_env=True, max_steps=3000)
        trainer.play(num_matches=100)

        dq_network = DQNKerasMem(action_wrapper=action_wrapper, state_builder=state_builder, nodes_layer1=256, nodes_layer2=256, nodes_layer3=256, nodes_layer4=256, learning_rate=0.005, gamma=0.90)
        agent = GenericAgent(dq_network, FrozenlakeReward())
        trainer = Trainer(env, agent, file_name=training_date + os.path.sep + "frozenlake_test_dqnKerasMem", save_every=1000, enable_save=True)
        # FrozenLake is solved when the agent is able to reach the end of the maze 100% of the times
        trainer.train(num_episodes=3000, reward_from_env=True, max_steps=3000)
        trainer.play(num_matches=100)

        dq_network = DDQNKeras(action_wrapper=action_wrapper, state_builder=state_builder, nodes_layer1=256, nodes_layer2=256, nodes_layer3=256, nodes_layer4=256, learning_rate=0.005, gamma=0.90)
        agent = GenericAgent(dq_network, FrozenlakeReward())
        trainer = Trainer(env, agent, file_name=training_date + os.path.sep + "frozenlake_test_ddqnKeras", save_every=1000, enable_save=True)
        # FrozenLake is solved when the agent is able to reach the end of the maze 100% of the times
        trainer.train(num_episodes=3000, reward_from_env=True, max_steps=3000)
        trainer.play(num_matches=100)

        dq_network = DQLTF(action_wrapper=action_wrapper, state_builder=state_builder, nodes_layer1=256, nodes_layer2=256, nodes_layer3=256, nodes_layer4=256, learning_rate=0.005, gamma=0.90)
        agent = GenericAgent(dq_network, FrozenlakeReward())
        trainer = Trainer(env, agent, file_name=training_date + os.path.sep + "frozenlake_test_dqltf", save_every=1000, enable_save=True)
        # FrozenLake is solved when the agent is able to reach the end of the maze 100% of the times
        trainer.train(num_episodes=3000, reward_from_env=True, max_steps=3000)
        trainer.play(num_matches=100)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)
