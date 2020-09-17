import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from base.savable import Savable 

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pickle
import os
from matplotlib.ticker import PercentFormatter
from tdd.reporter import Reporter as rp
from models.model_builder import ModelBuilder
from time import time

class Logger(Savable):
    def __init__(self, ep_total, agent_name, model_name, model_builder:ModelBuilder, action_wrapper_name, agent_action_size, agent_action_names, state_builder_name, reward_builder_name, env_name, is_episodic=True, render=True):
        #Training information
        self.agent_name = agent_name
        self.model_name = model_name
        self.model_builder = model_builder
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
        self.ep_victories = []
        self.ep_avg_victories = []

        # Agent Action count
        self.agent_action_names = agent_action_names
        self.agent_action_size = agent_action_size
        self.ep_agent_actions = [ [] for i in range(agent_action_size) ]

        # Play testing count
        self.play_ep_count = []
        self.play_rewards_avg = []
        self.play_match_count = []
        self.play_win_rates = []

        # Some agent info
        self.agent_info = []

        self.is_episodic = is_episodic

        self.avg_reward_graph = None 
        self.inst_reward_graph = None
        self.avg_steps_graph = None
        self.avg_winrate_graph = None
        
        #time and sps part
        self.training_start = time() 
        self.episode_duration_list = []
        self.episode_sps_list = []
        self.avg_sps_list = []
        self.episode_temp_start_time = 0

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

        self.ep_victories = []
        self.ep_avg_victories = []

    def record_episode_start(self):
        self.episode_temp_start_time = time()

    def record_episode(self, ep_reward, has_won, steps_count, agent_info, ep_actions):
        for i in range(self.agent_action_size):
            self.ep_agent_actions[i].append(ep_actions[i])

        self.ep_count += 1

        self.ep_rewards.append(ep_reward)
        self.ep_avg_rewards.append(sum(self.ep_rewards) / self.ep_count)
        
        self.ep_steps_count.append(steps_count)
        self.ep_avg_steps.append(sum(self.ep_steps_count) / self.ep_count)
        
        #time and sps stuff
        episode_duration = time() - self.episode_temp_start_time
        self.episode_duration_list.append(round(episode_duration, 1))
        self.episode_sps_list.append(round(steps_count / episode_duration, 2))
        self.avg_sps_list.append(round(sum(self.episode_sps_list) / self.ep_count, 2))

        if ep_reward > self.best_reward:
            self.best_reward = ep_reward
            self.best_reward_episode = self.ep_count

        if self.is_episodic:
            victory = 1 if has_won else 0
            self.ep_victories.append(victory)
            self.ep_avg_victories.append(sum(self.ep_victories)/ self.ep_count)

        self.agent_info.append(agent_info)

    def record_play_test(self, ep_count, play_rewards, play_victories, num_matches):
        self.play_ep_count.append(ep_count)
        self.play_match_count.append(num_matches)
        self.play_win_rates.append(play_victories/num_matches)
        self.play_rewards_avg.append(sum(play_rewards) / num_matches)

    def log_training_start_information(self):
        text = ("\n   Agent: {}\n".format(self.agent_name)
              + "   ActionWrapper: {}\n".format(self.action_wrapper_name)
              + "   StateBuilder: {}\n".format(self.state_builder_name)
              + "   RewardBuilder: {}\n".format(self.reward_builder_name)
              + "   Environment: {}\n".format (self.env_name)
              + "   Model: {}\n".format(self.model_name))

        # for idx, (layer) in enumerate(self.model_builder):
        #     if(layer['type'] == 'output'):
        #         text += "       Layer {}: type={} | length={} \n".format(idx+1, layer['type'], layer['length'])
        #     else:
        #         text += "       Layer {}: type={} | nodes={} \n".format(idx+1, layer['type'], layer['nodes'])

        self.training_report += text 

        rp.report(text)

    def log_ep_stats(self):
        if self.ep_count > 0:
            rp.report("Episode: {}/{} | Outcome: {} | Episode Avg. Reward: {:10.6f} | Episode Reward: {:10.6f} | Episode Steps: {:10.6f} | Best Reward was {} on episode: {} | Episode Duration (seconds): {} | Episode SPS: {} | SPS AVG: {} | Agent info: {}"
            .format(self.ep_count, self.ep_total, self.ep_victories[-1], self.ep_avg_rewards[-1], self.ep_rewards[-1], self.ep_steps_count[-1], self.best_reward, self.best_reward_episode, self.episode_duration_list[-1], self.episode_sps_list[-1], self.avg_sps_list[-1], self.agent_info[-1]))
        else:
            rp.report("There are no recorded episodes!")

    def log_train_stats(self):
        if self.ep_count > 0:
            text = ("\n"
            + "Current Reward Avg.: {}".format(sum(self.ep_rewards) / self.ep_count)
            + " Win rate: {:10.3f}%".format((sum(self.ep_victories)/ self.ep_count) * 100)
            + " Avg number of steps: {}".format(sum(self.ep_avg_steps)/ self.ep_count)
            + " Training Duration (seconds): {}".format(round(time() - self.training_start, 2))
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

    def generalized_curve_plot(self, to_plot, label, title):
        return self.__plot_curve(range(self.ep_count), to_plot, 'Episode Count',
                            label, title)

    def plot_average_reward_graph(self):
        # Plotting average reward graph
        return self.__plot_curve(range(self.ep_count), self.ep_avg_rewards, 'Episode Count',
                            'Avg. Reward', r'Reward avg. over training')

    def plot_average_steps_graph(self):
        # Plotting average steps graph
        return self.__plot_curve(range(self.ep_count), self.ep_avg_steps, 'Episode Count',
                            'Avg. Steps', r'Steps avg. over training')

    def plot_instant_reward_graph(self):
        # Plotting average reward graph
        return self.__plot_curve(range(self.ep_count), self.ep_rewards, 'Episode Count',
                            'Ep Reward', r'Episode Reward over training')

    def plot_win_rate_graph(self):
        # Plotting average reward graph
        return self.__plot_curve(range(self.ep_count), self.ep_avg_victories, 'Episode Count',
                            'Avg. Win Rate', r'Average Win Rate over training')

    def plot_win_rate_percentage_over_play_testing_graph(self):
        # Plotting win rate over play testing graph
        return self.__plot_bar(self.play_ep_count, [self.play_win_rates], ['Play'], 'Episode', 'Win rate (%)', 'Win rate percentage over play testing', format_percent=True)

    def plot_reward_average_over_play_testing_graph(self):
        # Plotting reward average over play testing graph
        return self.__plot_bar(self.play_ep_count, [self.play_rewards_avg], ['Play'], 'Episode', 'Reward avg.', 'Reward avg. over play testing')

    def save_extra(self, persist_path):
        if self.avg_reward_graph is None or self.avg_steps_graph is None or self.inst_reward_graph is None:
            self.render = False

            self.avg_reward_graph = self.plot_average_reward_graph()
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "avg_reward_graph.png")
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "avg_reward_graph.pdf")
            plt.close(self.avg_reward_graph)
            self.avg_reward_graph = None


            self.avg_steps_graph = self.plot_average_steps_graph() 
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "avg_steps_graph.png")
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "avg_steps_graph.pdf")
            plt.close(self.avg_steps_graph)
            self.avg_steps_graph = None


            self.inst_reward_graph = self.plot_instant_reward_graph()
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "inst_reward_graph.png")
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "inst_reward_graph.pdf")
            plt.close(self.inst_reward_graph)
            self.inst_reward_graph = None

            self.avg_winrate_graph = self.plot_win_rate_graph()
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "avg_winrate_graph.png")
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "avg_winrate_graph.pdf")
            plt.close(self.avg_winrate_graph)
            self.avg_winrate_graph = None

            temp_fig = self.generalized_curve_plot(self.episode_duration_list, "Episode Duration (Seconds)", "Per Episode Duration In Seconds")
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "ep_duration_graph.png")
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "ep_duration_graph.pdf")
            plt.close(temp_fig)

            temp_fig = self.generalized_curve_plot(self.episode_sps_list, "Episode SPS", "Per Episode SPS")
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "inst_ep_sps_graph.png")
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "inst_ep_sps_graph.pdf")
            plt.close(temp_fig)

            temp_fig = self.generalized_curve_plot(self.episode_sps_list, "Episode Avg. SPS", "Per Episode Avg. SPS")
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "avg_ep_sps_graph.png")
            plt.savefig(persist_path + os.path.sep + self.get_default_save_stamp() + "avg_ep_sps_graph.pdf")
            plt.close(temp_fig)

            # commenting this section to test whether or not the creation of action graphs impacts agent performance
            # # Populating self.agent_action_names with filler names if it wasn't provided by the agent's action wrapper
            # if self.agent_action_names == None:
            #     self.agent_action_names = []
            #     for i in range(self.agent_action_size):
            #         self.agent_action_names.append("Action "+str(i))
            # # Plotting the rate of occurrence of each action in a different graph
            # for i in range(self.agent_action_size):
            #     if self.agent_action_names != None:
            #         action_graph = self.generalized_curve_plot(self.ep_agent_actions[i], self.agent_action_names[i], "Plot for action " + self.agent_action_names[i])
            #         plt.savefig(persist_path + os.path.sep + "action_graphs" + os.path.sep + self.agent_action_names[i] + ".png")
            #     plt.close(action_graph)

            # # Plotting the rate of occurrence of all actions in one single graph
            # all_actions_graph = self.__plot_curves(range(self.ep_count), self.ep_agent_actions, 'Episode Count', "Actions per Ep.", self.agent_action_names, "Título")
            # plt.savefig(persist_path + os.path.sep + "action_graphs" + os.path.sep + "all_actions.png")
            # plt.close(all_actions_graph)

            self.render = True 

            with open(persist_path + os.path.sep + self.get_default_save_stamp() + "overall_report.txt", "w") as output:
                output.write(self.training_report)

    def __plot_curves(self, x, ys, x_label, y_label, y_labels, title):
        fig, ax = plt.subplots()
        for i in range(len(ys)):
            ax.plot(x, ys[i], label=y_labels[i])

        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
        ax.legend(bbox_to_anchor=(1.02,1), loc="upper left", ncol=2, prop={'size': 6})

        if self.render: 
            plt.ion()
            plt.show()
            plt.pause(0.001)

        return fig

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
