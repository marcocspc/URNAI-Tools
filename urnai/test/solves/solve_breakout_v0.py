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
from urnai.models.dqn_pytorch import DQNPytorch
from urnai.models.model_builder import ModelBuilder

def main(unused_argv):
    try:
        env = GymEnv(id="Breakout-v0", render=True)
        img = env.reset()

        action_wrapper = env.get_action_wrapper()
        state_builder = GymState(env.env_instance.observation_space.shape)

        helper = ModelBuilder()
        helper.add_convolutional_layer(filters=8, input_shape=(env.env_instance.observation_space.shape), dropout=0)
        helper.add_convolutional_layer(filters=4, dropout=0)
        # helper.add_input_layer(int(state_builder.get_state_dim()), 50)
        # helper.add_fullyconn_layer(50)
        helper.add_output_layer()

        # dq_network = DQNPytorch(action_wrapper=action_wrapper, state_builder=state_builder, build_model=helper.get_model_layout(),
        #                     gamma=0.99, learning_rate=0.001, epsilon_decay=0.99999, epsilon_min=0.01, memory_maxlen=100000)

        dq_network = DDQNKeras(action_wrapper=action_wrapper, state_builder=state_builder, build_model=helper.get_model_layout(),
                            gamma=0.99, learning_rate=0.001, learning_rate_min=0.0005, learning_rate_decay=0.99995, learning_rate_decay_ep_cutoff=500,
                            epsilon_decay=0.99999, epsilon_min=0.01, memory_maxlen=100000, min_memory_size=2000)

        #dq_network = PGKeras(action_wrapper, state_builder, learning_rate=0.001, gamma=0.99, build_model=helper.get_model_layout())

        agent = GenericAgent(dq_network, PureReward())

        trainer = Trainer(env, agent, save_path='urnai/models/saved', file_name="breakout-v0_test", 
                        save_every=100, enable_save=True, relative_path=True,
                        max_training_episodes=10, max_steps_training=700,
                        max_test_episodes=5, max_steps_testing=700)
        trainer.train()
        trainer.play()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)
