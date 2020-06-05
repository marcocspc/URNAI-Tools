import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from absl import app
from envs.gym_2048 import GymEnv2048
from trainers.trainer import Trainer
from agents.generic_agent import GenericAgent
from agents.actions.gym_wrapper import GymWrapper
from agents.rewards.gym import *
from agents.states.gym import Game2048State
from models.dql_keras import DQNKeras
from models.ddqn_keras import DDQNKeras
from models.model_builder import ModelBuilder

def main(unused_argv):
    try:
        env = GymEnv2048(_id="2048-v0")

        action_wrapper = env.get_action_wrapper()
        state_builder = Game2048State(env)

        helper = ModelBuilder()
        helper.add_input_layer(int(state_builder.get_state_dim()), nodes=24)
        helper.add_fullyconn_layer(nodes=24)
        helper.add_output_layer(action_wrapper.get_action_space_dim())

        dq_network = DDQNKeras(action_wrapper, state_builder, epsilon_decay=0.9997, build_model=helper.get_model_layout())

        agent = GenericAgent(dq_network, Game2048StdReward())

        # Using Trainer to train and play with our agent.
        #test_params = TestParams(num_matches=100, steps_per_test=250, max_steps=500, reward_threshold=1300)
        trainer = Trainer(env, agent, save_path='urnai/models/saved', file_name="game2048_ddqn_24x24", save_every=50, enable_save=True, relative_path=True)
        trainer.train(num_episodes=1000, max_steps=1000)
        trainer.play(num_matches=100, max_steps=1000, reward_from_agent=False)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)
