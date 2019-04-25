from absl import app
from envs.gym import GymEnv
from envs.trainer import Trainer
from agents.gym_agent import GymAgent
from agents.actions.gym_wrapper import GymWrapper
from agents.rewards.gym import FrozenlakeReward
from agents.states.gym import FrozenLakeState
from models.dql_working import DQNWorking

def main(unused_argv):
    trainer = Trainer()

    try:
        # Initializing our FrozenLake enviroment
        env = GymEnv(_id="FrozenLakeNotSlippery-v0")

        action_wrapper = GymWrapper(env)
        state_builder = FrozenLakeState()

        # Initializing a Deep Q-Learning model using our agent
        dq_network = DQNWorking(action_wrapper, state_builder, 'urnai/models/saved/frozenlake_dql_working')

        # Initializing our FrozenLake agent
        agent = GymAgent(dq_network, FrozenlakeReward())

        # Using Trainer to train and play with our agent.
        trainer.train(env, agent, num_episodes=10000, max_steps=100, save_steps=1000)
        trainer.play(env, agent, num_matches=100, max_steps=100)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)