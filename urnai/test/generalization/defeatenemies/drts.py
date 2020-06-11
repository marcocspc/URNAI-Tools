from urnai.scenarios.generalization.rts.defeatenemies import GeneralizedDefeatEnemiesScenario

episodes = 100
steps = 1000

env = GeneralizedDefeatEnemiesScenario(game = GeneralizedDefeatEnemiesScenario.GAME_DEEP_RTS, render = True)
action_wrapper = env.get_default_action_wrapper()

text = "Choose:\n"
cont = 0
for function in action_wrapper.get_actions():
    text += "\t{} - {}\n".format(action_wrapper.get_actions().index(function), function.__name__)

for ep in range(episodes):
    reward = 0
    done = False
    state = env.reset()
    print("Episode " + str(ep + 1))
    
    for step in range(steps):
            print("Step " + str(step + 1))

            print(env.env.render)
            
            action = None

            try:
                action = int(input(text))
            except ValueError:
                #No-action
                action = action_wrapper.noaction 

            if state is not None:
                action = action_wrapper.get_action(action, state)
            else:
                action = action_wrapper.noaction 

            state, reward, done = env.step(action)

            print("Reward: {r}".format(r=reward))

            if done: break
