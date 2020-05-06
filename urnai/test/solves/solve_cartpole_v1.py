import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

# # Forcing keras to use CPU instead of GPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from absl import app
from envs.gym import GymEnv
from trainers.trainer import Trainer
from trainers.trainer import TestParams
from agents.generic_agent import GenericAgent
from agents.actions.gym_wrapper import GymWrapper
from agents.rewards.default import PureReward
from agents.states.gym import PureState
from models.dql_keras_mem import DQNKerasMem
from models.model_builder import ModelBuilder

def main(unused_argv):
    try:
        env = GymEnv(id="CartPole-v1")

        action_wrapper = env.get_action_wrapper()
        state_builder = PureState(env.env_instance.observation_space)

        helper = ModelBuilder()
        helper.add_input_layer(int(state_builder.get_state_dim()))
        helper.add_fullyconn_layer(24)
        helper.add_fullyconn_layer(24)
        helper.add_output_layer(action_wrapper.get_action_space_dim())


        dq_network = DQNKerasMem(action_wrapper=action_wrapper, state_builder=state_builder, 
                                gamma=0.95, learning_rate=0.001, epsilon_decay=0.995, epsilon_min=0.01, 
                                build_model=helper.get_model_layout())

        # dq_network = PolicyGradientTF(action_wrapper, state_builder, 'urnai/models/saved/cartpole_v0_pg', learning_rate=0.01, gamma=0.9)

        agent = GenericAgent(dq_network, PureReward())

        # Cartpole-v1 is solved when avg. reward over 100 episodes is greater than or equal to 475
        #test_params = TestParams(num_matches=100, steps_per_test=100, max_steps=500, reward_threshold=500)
        trainer = Trainer(env, agent, save_path='urnai/models/saved', file_name="cartpole_v1", save_every=10, enable_save=True, relative_path=True)
        trainer.train(num_episodes=1000, max_steps=500, reward_from_env=True)
        trainer.play(num_matches=100, reward_from_env=True)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)
