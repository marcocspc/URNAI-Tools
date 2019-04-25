from absl import app
from envs.gym import GymEnv
from envs.trainer import Trainer
from agents.gym_agent import GymAgent
from agents.actions.gym_wrapper import GymWrapper
from agents.rewards.default import PureReward
from agents.states.gym import PureState
from models.dql_keras import DQNKeras

def main(unused_argv):
    trainer = Trainer()

    try:
        # Initializing our FrozenLake enviroment
        env = GymEnv(_id="CartPole-v1")

        # Initializing the action wrapper and state builder for our agent
        action_wrapper = GymWrapper(env)
        state_builder = PureState(env)

        # Initializing a Deep Q-Learning model
        dq_network = DQNKeras(action_wrapper, state_builder, 'urnai/models/saved/cartpole_dql_working', gamma=0.95, epsilon_decay=0.995, epsilon_min=0.1)

        # Initializing our FrozenLake agent
        agent = GymAgent(dq_network, PureReward())

        # Using Trainer to train and play with our agent.
        trainer.train(env, agent, num_episodes=4500, max_steps=1000, save_steps=1000)
        trainer.play(env, agent, num_matches=100, max_steps=1000)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)