import itertools

class Trainer():
    
    ## TODO: Add an option to play every x episodes, instead of just training non-stop
    ## TODO: Remove reward and win rate counts from this loop. They should be env-dependant. Consider
    ## creating a structure to allow developpers to choose which statistics they want to display while training.
    def train(self, env, agent, num_episodes=None, max_steps=None, save_steps=1000):
        num_episodes = num_episodes if num_episodes else float('inf')
        max_steps = max_steps if max_steps else float('inf')

        # List of rewards
        rewards = []

        #lista das medias de recompensa para imprimir o grafico
        reward_mean = [0]

        #lista do numero de vitorias
        victories = [0]
        victory_percentage = [0]

        print("Training...")

        try:
            for episode in itertools.count():
                if episode >= num_episodes:
                    break
                
                print("Episode " + str(episode + 1) + " out of " + str(num_episodes), end = "\r")

                env.start()
                agent.setup(env)

                # Reset the environment
                obs = env.reset()
                reward = 0
                done = False


                total_rewards = 0
                victory = False
                
                for step in itertools.count():
                    if step >= max_steps:
                        break
                    elif step == max_steps - 1:
                        done = True

                    action = agent.step(obs, reward, done)

                    if done:
                        victory = reward == 1
                        break
                    else:
                        obs, reward, done = env.step(action)
                        total_rewards += reward


                if episode % save_steps == 0:
                    agent.model.save()

                rewards.append(total_rewards)
                reward_mean.append(sum(rewards)/(episode + 1))

                if victory:
                    victories.append(1)
                else:
                    victories.append(0)
                victory_percentage.append(sum(victories)/(episode + 1))


        except KeyboardInterrupt:
            print()

        # Saving the model when the training is ended
        agent.model.save()

        print()
        print("Training ended!")
        print("Accumulated reward: " + str(sum(rewards) / num_episodes))
        print("Win rate: " + str(victory_percentage[-1]))
        print()


    def play(self, env, agent, num_matches, max_steps=None):
        num_matches = num_matches if num_matches else float('inf')
        max_steps = max_steps if max_steps else float('inf')

        rewards = []
        reward_mean = [0]

        victories = [0]
        victory_percentage = [0]

        print("Playing...")

        try:
            for match in itertools.count():
                if match >= num_matches:
                    break

                print("> Match " + str(match + 1) + " out of " + str(num_matches), end = "\r")

                env.start()
                agent.setup(env)

                # Reset the environment
                obs = env.reset()
                reward = 0
                done = False

                total_rewards = 0
                victory = False
                
                
                for step in itertools.count():
                    if step >= max_steps:
                        break

                    action = agent.play(obs)
                    
                    # If done (if we're dead) : finish episode
                    if done:
                        victory = reward == 1
                        break
                    else:
                        # Take the action (a) and observe the outcome state(s') and reward (r)
                        obs, reward, done = env.step(action)
                        total_rewards += reward


                rewards.append(total_rewards)
                reward_mean.append(sum(rewards)/(match + 1))

                if victory:
                    victories.append(1)
                else:
                    victories.append(0)
                victory_percentage.append(sum(victories)/(match + 1))

        except KeyboardInterrupt:
            print()

        print()
        print("Matches ended!")
        print("Accumulated reward: " + str(sum(rewards)/num_matches))
        print("Win rate: " + str(victory_percentage[-1]))
        print()

