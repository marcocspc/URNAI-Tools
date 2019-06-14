from absl import app
from envs.gym import GymEnv
from envs.trainer import Trainer
from agents.gym_agent import GymAgent
from agents.actions.gym_wrapper import GymWrapper
from agents.rewards.default import PureReward
from agents.states.gym import PureState
from models.pg_tf import PolicyGradientTF

def main(unused_argv):
    trainer = Trainer()

    try:
        env = GymEnv(_id="CartPole-v0")

        action_wrapper = GymWrapper(env)
        state_builder = PureState(env)

        dq_network = PolicyGradientTF(action_wrapper, state_builder, 'urnai/models/saved/cartpole_v0_pg')

        agent = GymAgent(dq_network, PureReward())

        # Using Trainer to train and play with our agent.
        trainer.train(env, agent, num_episodes=10000, max_steps=1000, save_steps=1000)
        trainer.play(env, agent, num_matches=100, max_steps=1000)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)