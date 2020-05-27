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
from agents.rewards.default import PureReward
from agents.states.gym import PureState
from agents.states.gym import GymState
from models.dql_keras_mem import DQNKerasMem
from models.dql_keras import DQNKeras
from models.pg_keras import PGKeras
from models.model_builder import ModelBuilder

def main(unused_argv):
    try:
        env = GymEnv(id="CartPole-v1")

        action_wrapper = env.get_action_wrapper()
        #state_builder = PureState(env.env_instance.observation_space)
        state_builder = GymState(env.env_instance.observation_space.shape[0])

        helper = ModelBuilder()
        helper.add_input_layer(int(state_builder.get_state_dim()))
        helper.add_fullyconn_layer(25)
        helper.add_fullyconn_layer(25)
        helper.add_output_layer(action_wrapper.get_action_space_dim())


        # dq_network = DQNKerasMem(action_wrapper=action_wrapper, state_builder=state_builder, 
        #                           gamma=0.99, learning_rate=0.001, epsilon_decay=0.995, epsilon_min=0.01, 
        #                           build_model=helper.get_model_layout(), memory_maxlen=500)

        dq_network = DQNKeras(action_wrapper=action_wrapper, state_builder=state_builder, 
                           gamma=0.99, learning_rate=0.001, epsilon_decay=0.9997, epsilon_min=0.01, memory_maxlen=50000, min_memory_size=1000)

        #dq_network = PGKeras(action_wrapper, state_builder, learning_rate=0.001, gamma=0.99, build_model=helper.get_model_layout())

        agent = GenericAgent(dq_network, PureReward())

        # Cartpole-v1 is solved when avg. reward over 100 episodes is greater than or equal to 475
        #test_params = TestParams(num_matches=100, steps_per_test=100, max_steps=500, reward_threshold=500)
        trainer = Trainer(env, agent, save_path='urnai/models/saved', file_name="cartpole_v1_dqn", save_every=500, enable_save=True, relative_path=True)
        trainer.train(num_episodes=1000, max_steps=500, reward_from_agent=True)
        trainer.play(num_matches=100, max_steps=500, reward_from_agent=True)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)
