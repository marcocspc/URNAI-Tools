import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from absl import app
from envs.gym import GymEnv
from envs.trainer import Trainer
from envs.trainer import TestParams
from agents.generic_agent import GenericAgent
from agents.actions.gym_wrapper import GymWrapper
from agents.rewards.default import PureReward
from agents.states.gym import PureState
from models.ql_table import QLearning
from models.dql_tf import DQLTF
from models.pg_tf import PolicyGradientTF

def main(unused_argv):
    trainer = Trainer()

    try:
        env = GymEnv(id="Taxi-v2")

        action_wrapper = GymWrapper(env)
        state_builder = PureState(env)

        # dq_network = PolicyGradientTF(action_wrapper, state_builder, 'urnai/models/saved/cartpole_v0_pg', learning_rate=0.01, gamma=0.95)
        dq_network = DQLTF(action_wrapper=action_wrapper, state_builder=state_builder, save_path='urnai/models/saved/cartpole_v0_dqltf', learning_rate=0.01)
        q_table = QLearning(action_wrapper, state_builder, save_path="models/saved/tax1-v2-Q-table", gamma=0.95, learning_rate=0.1)

        agent = GenericAgent(q_table, PureReward())

        # Taxi-v2 is solved when the agent is able to get an avg. reward of at least 8 over 100 matches (optimum is 8.46)
        test_params = TestParams(num_matches=100, steps_per_test=500, max_steps=200, reward_threshold=8.46)
        trainer.train(env, agent, num_episodes=10000, save_steps=100000, max_steps=200, test_params=test_params)
        trainer.play(env, agent, num_matches=100, max_steps=200)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
