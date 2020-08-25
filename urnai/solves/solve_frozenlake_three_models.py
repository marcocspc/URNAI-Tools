from absl import app
from urnai.envs.gym import GymEnv
from urnai.trainers.trainer import Trainer
from urnai.trainers.trainer import TestParams
from urnai.agents.generic_agent import GenericAgent
from urnai.agents.actions.gym_wrapper import GymWrapper
from urnai.agents.rewards.gym import FrozenlakeReward
from urnai.agents.states.gym import FrozenLakeState
from urnai.models.dqn_keras_mem import DQNKerasMem
from urnai.models.ddqn_keras import DDQNKeras 
from urnai.models.model_builder import ModelBuilder
from datetime import datetime

def main(unused_argv):
    try:
        env = GymEnv(id="FrozenLakeNotSlippery-v0")

        action_wrapper = env.get_action_wrapper() 
        state_builder = FrozenLakeState()

        training_date = str(datetime.now()).replace(" ","_").replace(":","_").replace(".","_")

        helper = ModelBuilder()
        helper.add_input_layer(int(state_builder.get_state_dim()))
        helper.add_fullyconn_layer(256)
        helper.add_fullyconn_layer(256)
        helper.add_fullyconn_layer(256)
        helper.add_fullyconn_layer(256)
        helper.add_output_layer(action_wrapper.get_action_space_dim())
        dq_network = DQNKerasMem(action_wrapper=action_wrapper, state_builder=state_builder, learning_rate=0.005, gamma=0.90, use_memory=False, per_episode_epsilon_decay = True, build_model=helper.get_model_layout())
        agent = GenericAgent(dq_network, FrozenlakeReward())
        trainer = Trainer(env, agent, file_name=training_date + os.path.sep + "frozenlake_test_dqnKerasMem", save_every=1000, enable_save=True)
        # FrozenLake is solved when the agent is able to reach the end of the maze 100% of the times
        trainer.train(num_episodes=3000, reward_from_env=True, max_steps=3000)
        trainer.play(num_matches=100)

        helper = ModelBuilder()
        helper.add_input_layer(int(state_builder.get_state_dim()))
        helper.add_fullyconn_layer(256)
        helper.add_fullyconn_layer(256)
        helper.add_fullyconn_layer(256)
        helper.add_fullyconn_layer(256)
        helper.add_output_layer(action_wrapper.get_action_space_dim())
        dq_network = DDQNKeras(action_wrapper=action_wrapper, state_builder=state_builder, learning_rate=0.005, gamma=0.90, use_memory=False, per_episode_epsilon_decay = True, build_model=helper.get_model_layout())
        agent = GenericAgent(dq_network, FrozenlakeReward())
        trainer = Trainer(env, agent, file_name=training_date + os.path.sep + "frozenlake_test_ddqnKeras", save_every=1000, enable_save=True)
        # FrozenLake is solved when the agent is able to reach the end of the maze 100% of the times
        trainer.train(num_episodes=3000, reward_from_env=True, max_steps=3000)
        trainer.play(num_matches=100)

        # helper = ModelBuilder()
        # helper.add_input_layer(int(state_builder.get_state_dim()))
        # helper.add_fullyconn_layer(256)
        # helper.add_fullyconn_layer(256)
        # helper.add_fullyconn_layer(256)
        # helper.add_fullyconn_layer(256)
        # helper.add_output_layer(action_wrapper.get_action_space_dim())
        # dq_network = DqlTfFlexible(action_wrapper=action_wrapper, state_builder=state_builder, learning_rate=0.005, gamma=0.9, build_model=helper.get_model_layout(), per_episode_epsilon_decay = True)
        # agent = GenericAgent(dq_network, FrozenlakeReward())
        # trainer = Trainer(env, agent, file_name=training_date + os.path.sep + "frozenlake_test_dql_flexible", save_every=1000, enable_save=True)
        # # FrozenLake is solved when the agent is able to reach the end of the maze 100% of the times
        # trainer.train(num_episodes=3000, reward_from_env=True, max_steps=3000)
        # trainer.play(num_matches=100)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)
