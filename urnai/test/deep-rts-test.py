import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from envs.deep_rts import DeepRTSEnv
import random

episodes = 100
steps = 100
drts = DeepRTSEnv(render=True)

for ep in range(episodes):
    print("Episode " + str(ep + 1))
    drts.reset()

    for step in range(steps):
        print("Step " + str(step + 1))
        state, done = drts.step(random.randint(0, 15))

        print("Current state: ")
        print(state)

        if done:
            break





