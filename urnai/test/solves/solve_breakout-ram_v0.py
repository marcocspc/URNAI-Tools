import os,sys
sys.path.insert(0, os.getcwd())

from absl import app
import gym
from urnai.envs.gym import GymEnv
from urnai.trainers.trainer import Trainer
from urnai.trainers.trainer import TestParams
from urnai.agents.generic_agent import GenericAgent
from urnai.agents.actions.gym_wrapper import GymWrapper
from urnai.agents.rewards.default import PureReward
from urnai.agents.states.gym import PureState
from urnai.agents.states.gym import GymState
from urnai.models.dqn_keras import DQNKeras
from urnai.models.ddqn_keras import DDQNKeras
from urnai.models.pg_keras import PGKeras
from urnai.models.model_builder import ModelBuilder

def main(unused_argv):
    try:
        env = GymEnv(id="Breakout-ram-v0", render=True)

        action_wrapper = env.get_action_wrapper()
        #state_builder = PureState(env.env_instance.observation_space)
        state_builder = GymState(env.env_instance.observation_space.shape[0])

        helper = ModelBuilder()
        helper.add_input_layer(int(state_builder.get_state_dim()), nodes=50)
        helper.add_fullyconn_layer(50)
        helper.add_output_layer(action_wrapper.get_action_space_dim())


        # dq_network = DQNKeras(action_wrapper=action_wrapper, state_builder=state_builder, 
        #                         gamma=0.99, learning_rate=0.001, epsilon_decay=0.9995, epsilon_min=0.01, 
        #                         build_model=helper.get_model_layout(), memory_maxlen=5000)

        dq_network = DDQNKeras(action_wrapper=action_wrapper, state_builder=state_builder, build_model=helper.get_model_layout(), use_memory=False,
                            gamma=0.99, learning_rate=0.001, epsilon_decay=0.999997, epsilon_min=0.01, memory_maxlen=100000, min_memory_size=2000)

        #dq_network = PGKeras(action_wrapper, state_builder, learning_rate=0.001, gamma=0.99, build_model=helper.get_model_layout())

        agent = GenericAgent(dq_network, PureReward())

        trainer = Trainer(env, agent, save_path='urnai/models/saved', file_name="breakout-ram-v0_pg_50x50", 
                        save_every=100, enable_save=True, relative_path=True,
                        max_training_episodes=10000, max_steps_training=1800,
                        max_test_episodes=100, max_steps_testing=1800)
        trainer.train()
        trainer.play()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)
