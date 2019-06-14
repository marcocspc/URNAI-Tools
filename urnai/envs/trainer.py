import itertools


class Trainer():

    ## TODO: Add an option to play every x episodes, instead of just training non-stop
    ## TODO: Remove reward and win rate counts from this loop. They should be env-dependant. Consider
    ## creating a structure to allow developpers to choose which statistics they want to display while training.
    def train(self, env, agent, num_episodes=float('inf'), max_steps=float('inf'), save_steps=1000):
        # List of rewards
        rewards = []

        # lista do numero de vitorias
        victories = [0]
        victory_percentage = [0]

        print("Training...")

        for episode in itertools.count():
            if episode >= num_episodes:
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

                is_last_step = step == max_steps - 1

                action = agent.step(obs, reward, done)
                obs, reward, done = env.step(action)
                agent.learn(obs, reward, done, is_last_step)

                ep_reward += reward

                if done or is_last_step:
                    print("Episode: {}/{} | Avg. reward: {}".format(episode + 1, num_episodes, sum(rewards) / (episode + 1)), end="\r")
                    victory = reward == 1
                    break

            if episode > 0 and episode % save_steps == 0:
                agent.model.save()
                self.printPerformance(rewards, num_episodes, victory_percentage)

            rewards.append(ep_reward)

            if victory:
                victories.append(1)
            else:
                victories.append(0)
            victory_percentage.append(sum(victories) / (episode + 1))

        # Saving the model when the training is ended
        #agent.model.save()

        self.printPerformance(rewards, num_episodes, victory_percentage)

    def play(self, env, agent, num_matches, max_steps=float('inf')):
        rewards = []

        victories = [0]
        victory_percentage = [0]

        print("Playing...")

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

                # If done (if we're dead) : finish episode
                if done:
                    victory = reward == 1
                    break

            rewards.append(ep_reward)

            if victory:
                victories.append(1)
            else:
                victories.append(0)
            victory_percentage.append(sum(victories) / (match + 1))

            print("Match: {}/{} | Avg. reward: {}".format(match + 1, num_matches, sum(rewards) / (match + 1)), end="\r")

        self.printPerformance(rewards, num_matches, victory_percentage)

    def printPerformance(self, rewards, num_matches, victory_percentage):
        print()
        print("Matches ended!")
        print("Average reward: " + str(sum(rewards) / num_matches))
        print("Win rate: " + str(victory_percentage[-1]))
        print()
