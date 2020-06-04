from urnai.scenarios.generalization.rts.defeatenemies import GeneralizedDefeatEnemiesScenario

episodes = 100
steps = 1000

env = GeneralizedDefeatEnemiesScenario(game = GeneralizedDefeatEnemiesScenario.GAME_STARCRAFT_II, render = True)
action_wrapper = env.get_default_action_wrapper()

for ep in range(episodes):
    reward = 0
    done = False
    state = env.reset()
    print("Episode " + str(ep + 1))
    
    for step in range(steps):
            print("Step " + str(step + 1))
            
            text = '''
                Choose:
                    1 - Left 
                    2 - Right 
                    3 - Up 
                    4 - Down 
                    5 - Attack Nearest Unit 
                    6 - No-Op
            '''

            action = None

            try:
                action = int(input(text))
            except ValueError:
                action = 6

            action -= 1
            action = action_wrapper.get_action(action, state)

            state, reward, done = env.step(action)

            print("Reward: {r}".format(r=reward))

            if done: break
