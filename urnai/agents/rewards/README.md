# Reward Builder

Reinforcement Learning Agents have no prior knowledge about how good a particular action is on any given state (in contrast to Supervised Learning which has labeled states). Therefore, in order to learn the Agent needs some sort of direction, a guiding function. This is exactly the role of a Reward Function.

During each learning cycle the Agent performs an action in the environment, and receives back both a state observation and a reward. This reward is generally a real number, such as 1, 10.34, -3, etc. These numbers should represent if the action taken by the Agent was good, or bad, and how good or bad it was.

In order for us to be able to control learning, we have to design these reward functions, create rules and reward values that should give the Agent a positive feedback whenever it does something deemed good, and maybe punish it when it does something deemed bad. In a video-game environment, these rewards generally tend to try and make the Agent learn how to win at the game, or get a higher score.

## Creating Reward Builders

In URNAI the base class for a reward function is called [RewardBuilder](./abreward.py). This class is very simple, as the principle of a RewardBuilder should be very straight forward. It has an abstract method called _get_reward_, defined below, that implements the calculation of a reward for any given game state.

```python
@abstractmethod
def get_reward(self, obs, reward, done) -> Reward: ...
```

this method receives three parameters: the raw environment observation (obs), the default enviroment reward (reward), and a bool representing if this is the last step of the game, in other words if the game is ending (done).

It will then implement the reward function whichever way you desire, either using the direct reward from the game environment, or creating your own custom reward using information from _obs_.

To see a very simple but complete Reward Builder, take a look at our default [PureReward class](./default.py). It simply takes the _reward_ parameter passed to it and returns it in _get_reward_.

More complex Reward Builders can be seen in our files for [VizDoom rewards](./vizdoom.py) and for [StarCraft II rewards](./sc2.py).

An important internal state control that Reward Builders sometimes have to maintain is whether or not to reset themselves when an episode has ended. Most of our SC2 rewards do this, and it can be achieved by checking if done is true, and then reseting the reward at the end of the _get_reward_ method, like so:

```python
if done:
    self.reset()
```

It is up to you to figure out the best way to reset your Reward Builder, but in any case the base reset method is implemented inside the RewardBuilder class in a way so that it doesn't do anything by default, allowing you to create reward functions without caring about the reset, unless it is necessary.
