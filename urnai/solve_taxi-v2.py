from absl import app
from envs.gym import GymEnv
from envs.trainer import Trainer
from agents.gym_agent import GymAgent
from agents.actions.gym_wrapper import GymWrapper
from agents.rewards.default import PureReward
from agents.states.gym import PureState
from models.ql_table import QLearning


def main(unused_argv):
    trainer = Trainer()

    try:
        # Initializing our FrozenLake enviroment
        env = GymEnv(_id="Taxi-v2")

        action_wrapper = GymWrapper(env)
        state_builder = PureState(env)

        # Initializing a Q-Table model using our agent
        q_table = QLearning(action_wrapper, state_builder, save_path="models/saved/tax1-v2-Q-table")

        # Initializing our FrozenLake agent
        agent = GymAgent(q_table, PureReward())

        # Using Trainer to train and play with our agent.
        trainer.train(env, agent, num_episodes=100000, save_steps=100)
        trainer.play(env, agent, num_matches=100, max_steps=100)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
