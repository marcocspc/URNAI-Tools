import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from base.savable import Savable 

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from matplotlib.ticker import PercentFormatter
from urnai.tdd.reporter import Reporter as rp

class Logger(Savable):
    def __init__(self, ep_total, agent_name, model_name, action_wrapper_name, state_builder_name, reward_builder_name, env_name, is_episodic=True, render=True):
        #Training information
        self.agent_name = agent_name
        self.model_name = model_name
        self.action_wrapper_name = action_wrapper_name
        self.state_builder_name = state_builder_name
        self.reward_builder_name = reward_builder_name
        self.env_name = env_name

        # Episode count
        self.ep_count = 0
        self.ep_total = ep_total

        # Reward count
        self.best_reward = -999999
        self.best_reward_episode = -1
        self.ep_rewards = []
        self.ep_avg_rewards = []

        # Steps count
        self.ep_steps_count = []
        self.ep_avg_steps = []

        # Win rate count
        self.victories = 0

        # Play testing count
        self.play_ep_count = []
        self.play_rewards_avg = []
        self.play_match_count = []
        self.play_win_rates = []

        # Some agent info
        self.agent_info = []

        self.is_episodic = is_episodic

        self.avg_reward_graph = None 
        self.avg_steps_graph = None

        #Training report
        self.training_report = ""

        self.render = render

        self.log_training_start_information()

    def reset(self):
        self.ep_count = 0

        self.ep_rewards = []
        self.ep_avg_rewards = []

        self.ep_steps_count = []
        self.ep_avg_steps = []

        self.victories = 0

    def record_episode(self, ep_reward, has_won, steps_count, agent_info):
        self.ep_count += 1

        self.ep_rewards.append(ep_reward)
        self.ep_avg_rewards.append(sum(self.ep_rewards) / self.ep_count)
        
        self.ep_steps_count.append(steps_count)
        self.ep_avg_steps.append(sum(self.ep_steps_count) / self.ep_count)

        if ep_reward > self.best_reward:
            self.best_reward = ep_reward
            self.best_reward_episode = self.ep_count

        if self.is_episodic and has_won:
            self.victories += 1

        self.agent_info.append(agent_info)

    def record_play_test(self, ep_count, play_rewards, play_victories, num_matches):
        self.play_ep_count.append(ep_count)
        self.play_match_count.append(num_matches)
        self.play_win_rates.append(play_victories/num_matches)
        self.play_rewards_avg.append(sum(play_rewards) / num_matches)

    def log_training_start_information(self):
        text = ("    Agent: {}\n".format(self.agent_name)
                + "        Model: {}\n".format(self.model_name)
                + "        ActionWrapper: {}\n".format(self.action_wrapper_name)
                + "        StateBuilder: {}\n".format(self.state_builder_name)
                + "        RewardBuilder: {}\n".format(self.reward_builder_name)
                + "    Environment: {}\n".format (self.env_name))

        self.training_report += text 

        rp.report(text)

    def log_ep_stats(self):
        if self.ep_count > 0:
            rp.report("Episode: {}/{} | Episode Avg. Reward: {:10.6f} | Episode Steps: {:10.6f} | Best Reward was {} on episode: {} | Agent info: {}"
            .format(self.ep_count, self.ep_total, self.ep_avg_rewards[-1], self.ep_steps_count[-1], self.best_reward, self.best_reward_episode, self.agent_info[-1]), end="\r")
        else:
            rp.report("There are no recorded episodes!")

    def log_train_stats(self):
        if self.ep_count > 0:
            text = ("\n"
            + "Current Reward Avg.: {}".format(sum(self.ep_rewards) / self.ep_count)
            + " Win rate: {:10.3f}%".format((self.victories / self.ep_count) * 100)
            + " Avg number of steps: {}".format(sum(self.ep_avg_steps)/ self.ep_count)
            + "\n")

            self.training_report += text 

            rp.report(text)
        else:
            rp.report("There are no recorded episodes!")
    
    def plot_train_stats(self):
        self.plot_average_reward_graph()

        self.plot_average_steps_graph()
   
        if len(self.play_ep_count) > 0:
            self.plot_win_rate_percentage_over_play_testing_graph()
            self.plot_reward_average_over_play_testing_graph()

    def plot_average_reward_graph(self):
        # Plotting average reward graph
        return self.__plot_curve(range(self.ep_count), self.ep_avg_rewards, 'Episode Count',
                            'Avg. Reward', r'Reward avg. over training')

    def plot_average_steps_graph(self):
        # Plotting average steps graph
        return self.__plot_curve(range(self.ep_count), self.ep_avg_steps, 'Episode Count',
                            'Avg. Steps', r'Steps avg. over training')

    def plot_win_rate_percentage_over_play_testing_graph(self):
        # Plotting win rate over play testing graph
        return self.__plot_bar(self.play_ep_count, [self.play_win_rates], ['Play'], 'Episode', 'Win rate (%)', 'Win rate percentage over play testing', format_percent=True)

    def plot_reward_average_over_play_testing_graph(self):
        # Plotting reward average over play testing graph
        return self.__plot_bar(self.play_ep_count, [self.play_rewards_avg], ['Play'], 'Episode', 'Reward avg.', 'Reward avg. over play testing')

    def save_extra(self, persist_path):
        if self.avg_reward_graph is None or self.avg_steps_graph is None:
             self.render = False

             self.avg_reward_graph = self.plot_average_reward_graph()
             plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "avg_reward_graph_bar.png")
             plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "avg_reward_graph_bar.pdf")
             plt.close(self.avg_reward_graph)
             self.avg_reward_graph = None


             self.avg_steps_graph = self.plot_average_steps_graph() 
             plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "avg_steps_graph_bar.png")
             plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "avg_steps_graph_bar.pdf")
             plt.close(self.avg_steps_graph)
             self.avg_steps_graph = None

             self.render = True 

             with open(persist_path + os.path.sep + self.get_default_save_stamp() + "overall_report.txt", "w") as output:
                 output.write(self.training_report)

    def __plot_curve(self, x, y, x_label, y_label, title):
        fig, ax = plt.subplots()
        ax.plot(x, y)

        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()

        if self.render: 
            plt.ion()
            plt.show()
            plt.pause(0.001)

        return fig

    def __plot_bar(self, x_values, y_bars, bar_labels, x_label, y_label, title, width=0.2, format_percent=False, percent_scale=1):
        fig, ax = plt.subplots()

        x = np.arange(len(x_values))  # List of label locations for the x values
        bar_count = len(y_bars)
        min_width = x - width / bar_count
        max_width = x + width / bar_count
        for bar, bar_label, bar_idx in zip(y_bars, bar_labels, range(bar_count)):
            bar_width = x if bar_count == 1 else self.__lerp(min_width, max_width, bar_idx / (bar_count - 1))
            rects = ax.bar(bar_width, bar, width, label=bar_label)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_values)
        ax.legend()

        if format_percent:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=percent_scale))

        if self.render: 
            plt.ion()
            plt.show()
            plt.pause(0.001)

        return fig

    def __lerp(self, a, b, t):
        return (1 - t) * a + t * b
