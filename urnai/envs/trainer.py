import itertools
from utils.logger import Logger

class TestParams():
    def __init__(self, num_matches, steps_per_test, max_steps=float('inf')):
        self.num_matches = num_matches
        self.test_steps = steps_per_test
        self.max_steps = max_steps
        self.current_ep_count = 0
        self.logger = None


class Trainer():
    ## TODO: Add an option to play every x episodes, instead of just training non-stop
    def train(self, env, agent, num_episodes=float('inf'), max_steps=float('inf'), save_steps=1000, enable_save=True, test_params: TestParams = None):
        print("> Training")

        logger = Logger(num_episodes)

        if test_params != None:
            test_params.logger = logger

        for episode in itertools.count():
            if episode >= num_episodes:
                break

            env.start()

            # Reset the environment
            obs = env.reset()
            step_reward = 0
            done = False
            agent.reset()

            ep_reward = 0
            victory = False

            for step in itertools.count():
                if step >= max_steps:
                    break

                is_last_step = step == max_steps - 1

                action = agent.step(obs, step_reward, done)
                obs, step_reward, done = env.step(action)
                agent.learn(obs, step_reward, done, is_last_step)

                ep_reward += step_reward

                if done or is_last_step:
                    victory = step_reward == 1
                    logger.record_episode(ep_reward, victory, step + 1)
                    break
                    
            logger.log_ep_stats()
            if enable_save and episode > 0 and episode % save_steps == 0:
                agent.model.save()

            if test_params != None and episode % test_params.test_steps == 0:
                test_params.current_ep_count = episode
                self.play(env, agent, test_params.num_matches, test_params.max_steps, test_params)

        # Saving the model when the training is ended
        if enable_save:
            agent.model.save()
        logger.log_train_stats()
        logger.plot_train_stats(agent)


    def play(self, env, agent, num_matches, max_steps=float('inf'), test_params=None):
        print()
        print()
        print("> Playing")

        logger = Logger(num_matches)

        for match in itertools.count():
            if match >= num_matches:
                break

            env.start()

            # Reset the environment
            obs = env.reset()
            reward = 0
            done = False
            agent.reset()

            ep_reward = 0
            victory = False

            for step in itertools.count():
                if step >= max_steps:
                    break

                action = agent.play(obs)
                # Take the action (a) and observe the outcome state(s') and reward (r)
                obs, reward, done = env.step(action)
                ep_reward += reward

                is_last_step = step == max_steps - 1
                # If done (if we're dead) : finish episode
                if done or is_last_step:
                    victory = reward == 1
                    logger.record_episode(ep_reward, victory, step + 1)
                    break

            logger.log_ep_stats()

        if test_params != None:
            test_params.logger.record_play_test(test_params.current_ep_count, logger.ep_rewards, logger.victories, num_matches)
            print()
        else:
            # Only logs train stats if this is not a test, to avoid cluttering the interface with info
            logger.log_train_stats()
