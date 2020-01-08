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
from agents.rewards.gym import FrozenlakeReward
from agents.states.gym import FrozenLakeState
from models.dql_tf import DQLTF
from models.pg_tf import PolicyGradientTF

def main(unused_argv):
    trainer = Trainer()

    try:
        env = GymEnv(id="FrozenLakeNotSlippery-v0")

        action_wrapper = GymWrapper(env)
        state_builder = FrozenLakeState()

        # dq_network = DQLTF(action_wrapper, state_builder, 'urnai/models/saved/frozenlake_dql_working', learning_rate=0.0008, gamma=0.9)
        dq_network = PolicyGradientTF(action_wrapper, state_builder, 'urnai/models/saved/frozenlake_v0_pg', learning_rate=0.01, gamma=0.9)

        agent = GenericAgent(dq_network, FrozenlakeReward())

        # FrozenLake is solved when the agent is able to reach the end of the maze 100% of the times
        test_params = TestParams(num_matches=100, steps_per_test=100, max_steps=30, reward_threshold=1)
        trainer.train(env, agent, num_episodes=3000, max_steps=20, save_steps=1000, test_params=test_params, enable_save=False)
        trainer.play(env, agent, num_matches=100, max_steps=20)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)
