from absl import app
from envs.gym_2048 import GymEnv2048
from envs.trainer import Trainer
from envs.trainer import TestParams
from agents.generic_agent import GenericAgent
from agents.actions.gym_wrapper import GymWrapper
from agents.rewards.gym import Game2048Reward
from agents.states.gym import Game2048State
from models.dql_keras import DQNKeras
# from models.dql_keras_mem import DQNKerasMem
from models.dql_tf import DQLTF
from models.pg_tf import PolicyGradientTF

def main(unused_argv):
    trainer = Trainer()

    try:
        env = GymEnv2048(_id="2048-v0")

        action_wrapper = GymWrapper(env)
        state_builder = Game2048State(env)

        dq_network = DQLTF(action_wrapper, state_builder, 'urnai/models/saved/game2048_dqltf_1212')
        # dq_network = DQNKeras(action_wrapper, state_builder, 'urnai/models/saved/game2048_divReward_dqnkeras_mem50000_1212', gamma=0.95, epsilon_decay=0.995, epsilon_min=0.1, batch_size=2)
        # dq_network = DQNKerasMem(action_wrapper, state_builder, 'urnai/models/saved/game2048_divReward_dqnkerasmem_1616', gamma=0.95, epsilon_decay=0.995, epsilon_min=0.1, batch_size=6)
        # dq_network = PolicyGradientTF(action_wrapper, state_builder, 'urnai/models/saved/game2048_pg', learning_rate=0.01, gamma=0.95)

        agent = GenericAgent(dq_network, Game2048Reward(sparce=False))

        # Using Trainer to train and play with our agent.
        test_params = TestParams(num_matches=100, steps_per_test=250, max_steps=500, reward_threshold=1300)
        trainer.train(env, agent, num_episodes=2000, max_steps=500, save_steps=500, test_params=test_params, enable_save=True, reward_from_builder=False)
        trainer.play(env, agent, num_matches=100, max_steps=500)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)