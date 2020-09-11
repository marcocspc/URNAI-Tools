import sys,os,inspect
import itertools
import time
import numpy as np
from utils.logger import Logger
from base.savable import Savable 
from tdd.reporter import Reporter as rp
from datetime import datetime

class TestParams():
    def __init__(self, num_matches, steps_per_test, max_steps=float('inf'), reward_threshold=None):
        self.num_matches = num_matches
        self.test_steps = steps_per_test
        self.max_steps = max_steps
        self.current_ep_count = 0
        self.logger = None
        self.reward_threshold = reward_threshold

class Trainer(Savable):
    ## TODO: Add an option to play every x episodes, instead of just training non-stop

    def __init__(self, env, agent, save_path=os.path.expanduser("~") + os.path.sep + "urnai_saved_traingings", file_name=str(datetime.now()).replace(" ","_").replace(":","_").replace(".","_"), enable_save=False, save_every=10, relative_path=False, debug_level=0):
        super().__init__()
        self.setup(env, agent, save_path, file_name, enable_save, save_every, relative_path, debug_level)

    def setup(self, env, agent, save_path=os.path.expanduser("~") + os.path.sep + "urnai_saved_traingings", file_name=str(datetime.now()).replace(" ","_").replace(":","_").replace(".","_"), enable_save=False, save_every=10, relative_path=False, debug_level=0):
        self.env = env
        self.agent = agent
        self.save_path = save_path
        self.file_name = file_name 
        self.enable_save = enable_save
        self.save_every = save_every
        self.relative_path = relative_path
        self.pickle_black_list.append("agent")
        rp.VERBOSITY_LEVEL = debug_level

        self.logger = Logger(0, self.agent.__class__.__name__, self.agent.model.__class__.__name__, self.agent.model.build_model, self.agent.action_wrapper.__class__.__name__, self.agent.action_wrapper.get_action_space_dim(), self.agent.action_wrapper.get_named_actions(), self.agent.state_builder.__class__.__name__, self.agent.reward_builder.__class__.__name__, self.env.__class__.__name__) 

        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)
        parentdir = os.path.dirname(parentdir)
        if(relative_path):
            self.full_save_path = parentdir + os.path.sep + self.save_path + os.path.sep + self.file_name
        else:
            self.full_save_path = self.save_path + os.path.sep + self.file_name 
        
        self.full_save_play_path = self.full_save_path + os.path.sep + "play_files"

        if self.enable_save and os.path.exists(self.full_save_path):
            rp.report("WARNING! Loading training from " + self.full_save_path + " with SAVING ENABLED.")
            self.load(self.full_save_path)
        elif self.enable_save:
            rp.report("WARNING! Starting new training on " + self.full_save_path + " with SAVING ENABLED.")
            os.makedirs(self.full_save_path)
            os.makedirs(self.full_save_path + os.path.sep + "action_graphs")
            os.makedirs(self.full_save_play_path)
            os.makedirs(self.full_save_play_path + os.path.sep + "action_graphs")
        else:
            rp.report("WARNING! Starting new training WITHOUT SAVING PROGRESS.")


    def train(self, num_episodes=float('inf'), max_steps=float('inf'), test_params: TestParams = None, reward_from_agent = True):
        start_time = time.time()
        
        rp.report("> Training")
        if self.logger.ep_count == 0:
            self.logger = Logger(num_episodes, self.agent.__class__.__name__, self.agent.model.__class__.__name__, self.agent.model.build_model, self.agent.action_wrapper.__class__.__name__, self.agent.action_wrapper.get_action_space_dim(), self.agent.action_wrapper.get_named_actions(), self.agent.state_builder.__class__.__name__, self.agent.reward_builder.__class__.__name__, self.env.__class__.__name__) 

        if test_params != None:
            test_params.logger = self.logger

        for episode in itertools.count():
            if episode >= num_episodes:
                break

            self.env.start()

            # Reset the environment
            obs = self.env.reset()
            step_reward = 0
            done = False
            # Passing the episode to the agent reset, so that it can be passed to model reset
            # Allowing the model to track the episode number, and decide if it should diminish the
            # Learning Rate, depending on the currently selected strategy.
            self.agent.reset(episode)

            ep_reward = 0
            victory = False

            ep_actions = np.zeros(self.agent.action_wrapper.get_action_space_dim())

            for step in itertools.count():
                if step >= max_steps:
                    break
                
                # Choosing an action and passing it to our env.step() in order to act on our environment
#                try:
                action = self.agent.step(obs, step_reward, done)
#                except ValueError:
#                    np.set_printoptions(threshold=sys.maxsize)
#                    print(obs)

                obs, default_reward, done = self.env.step(action)

                is_last_step = step == max_steps - 1
                done = done or is_last_step

                # Checking whether or not to use the reward from the reward builder so we can pass that to the agent
                if reward_from_agent:
                    step_reward = self.agent.get_reward(obs, default_reward, done)
                else:
                    step_reward = default_reward

                # Making the agent learn
                self.agent.learn(obs, step_reward, done)

                # Adding our step reward to the total count of the episode's reward
                ep_reward += step_reward

                ep_actions[self.agent.previous_action] += 1

                if done:
                    victory = default_reward == 1
                    agent_info = {
                            "Learning rate" : self.agent.model.learning_rate,
                            "Gamma" : self.agent.model.gamma,
                            "Epsilon" : self.agent.model.epsilon_greedy,
                            }
                    self.logger.record_episode(ep_reward, victory, step + 1, agent_info, ep_actions)
                    break
            
            self.logger.log_ep_stats()
            if self.enable_save and episode > 0 and episode % self.save_every == 0:
                #self.logger.log_ep_stats()
                self.save(self.full_save_path)
            #if enable_save and episode > 0 and episode % save_steps == 0:
                #self.save(self.full_save_path)

            if test_params != None and episode % test_params.test_steps == 0 and episode != 0:
                test_params.current_ep_count = episode
                self.play(test_params.num_matches, test_params.max_steps, test_params)

                # Stops training if reward threshold was reached in play testing
                if test_params.reward_threshold != None and test_params.reward_threshold <= test_params.logger.play_rewards_avg[-1]:
                    rp.report("> Reward threshold was reached!")
                    rp.report("> Stopping training")
                    break

        end_time = time.time()
        rp.report("\n> Training duration: {} seconds".format(end_time - start_time))

        self.logger.log_train_stats()
        self.logger.plot_train_stats()
        # Saving the model when the training has ended
        if self.enable_save:
            self.save(self.full_save_path)


    def play(self, num_matches, max_steps=float('inf'), test_params=None, reward_from_agent = True):
        rp.report("\n\n> Playing")

        self.logger = Logger(num_matches, self.agent.__class__.__name__, self.agent.model.__class__.__name__, self.agent.model.build_model, self.agent.action_wrapper.__class__.__name__, self.agent.action_wrapper.get_action_space_dim(), self.agent.action_wrapper.get_named_actions(), self.agent.state_builder.__class__.__name__, self.agent.reward_builder.__class__.__name__, self.env.__class__.__name__) 

        for match in itertools.count():
            if match >= num_matches:
                break

            self.env.start()

            # Reset the environment
            obs = self.env.reset()
            step_reward = 0
            done = False
            # Passing the episode to the agent reset, so that it can be passed to model reset
            # Allowing the model to track the episode number, and decide if it should diminish the
            # Learning Rate, depending on the currently selected strategy.
            self.agent.reset(match)

            ep_reward = 0
            victory = False

            ep_actions = np.zeros(self.agent.action_wrapper.get_action_space_dim())

            for step in itertools.count():
                if step >= max_steps:
                    break

                action = self.agent.play(obs)
                # Take the action (a) and observe the outcome state(s') and reward (r)
                obs, default_reward, done = self.env.step(action)

                if reward_from_agent:
                    step_reward = self.agent.get_reward(obs, default_reward, done)
                else:
                    step_reward = default_reward

                ep_reward += step_reward

                ep_actions[self.agent.previous_action] += 1

                is_last_step = step == max_steps - 1
                done = done or is_last_step
                # If done: finish episode
                if done:
                    victory = default_reward == 1
                    agent_info = {
                            "Learning rate" : self.agent.model.learning_rate,
                            "Gamma" : self.agent.model.gamma,
                            "Epsilon" : self.agent.model.epsilon_greedy,
                            }
                    self.logger.record_episode(ep_reward, victory, step + 1, agent_info, ep_actions)
                    break

            self.logger.log_ep_stats()

        if test_params != None:
            test_params.logger.record_play_test(test_params.current_ep_count, self.logger.ep_rewards, self.logger.victories, num_matches)
        else:
            # Only logs train stats if this is not a test, to avoid cluttering the interface with info
            self.logger.log_train_stats()

        #We need to save playing status as well 
        if self.enable_save:
            self.logger.save(self.full_save_play_path)
            rp.save(self.full_save_play_path)
            #self.save(self.full_save_play_path)

    def save_extra(self, save_path):
        self.env.save(save_path)
        self.agent.save(save_path)
        self.logger.save(save_path)
        rp.save(save_path)

    def load_extra(self, save_path):
        self.agent.load(save_path)
        self.env.load(save_path)
        self.logger.load(save_path)
        rp.load(save_path)
