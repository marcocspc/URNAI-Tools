# Forcing keras to use CPU instead of GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from absl import app
from envs.gym import GymEnv
from envs.trainer import Trainer
from envs.trainer import TestParams
from agents.generic_agent import GenericAgent
from agents.actions.gym_wrapper import GymWrapper
from agents.rewards.default import PureReward
from agents.states.gym import PureState
from models.dql_keras import DQNKeras

def main(unused_argv):
    trainer = Trainer()

    try:
        env = GymEnv(_id="CartPole-v1")

        action_wrapper = GymWrapper(env)
        state_builder = PureState(env)

        dq_network = DQNKeras(action_wrapper, state_builder, 'urnai/models/saved/cartpole_dql_working',
                                gamma=0.99, learning_rate=0.001, epsilon_decay=0.995, epsilon_min=0.1, batch_size=32)

        agent = GenericAgent(dq_network, PureReward())

        # Cartpole-v1 is solved when avg. reward over 100 episodes is greater than or equal to 475
        test_params = TestParams(num_matches=100, steps_per_test=200, max_steps=500)
        trainer.train(env, agent, num_episodes=1250, max_steps=500, save_steps=1000, test_params=test_params, enable_save=False)
        trainer.play(env, agent, num_matches=100, max_steps=500)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)