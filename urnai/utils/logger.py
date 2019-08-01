import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

class Logger():
    def __init__(self, ep_total, is_episodic=True):
        # Episode count
        self.ep_count = 0
        self.ep_total = ep_total

        # Reward count
        self.best_reward = -999999
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

        self.is_episodic = is_episodic
    
    def reset(self):
        self.ep_count = 0

        self.ep_rewards = []
        self.ep_avg_rewards = []

        self.ep_steps_count = []
        self.ep_avg_steps = []

        self.victories = 0

    def record_episode(self, ep_reward, has_won, steps_count):
        self.ep_count += 1

        self.ep_rewards.append(ep_reward)
        self.ep_avg_rewards.append(sum(self.ep_rewards) / self.ep_count)
        
        self.ep_steps_count.append(steps_count)
        self.ep_avg_steps.append(sum(self.ep_steps_count) / self.ep_count)

        if ep_reward > self.best_reward:
            self.best_reward = ep_reward

        if self.is_episodic and has_won:
            self.victories += 1

    def record_play_test(self, ep_count, play_rewards, play_victories, num_matches):
        self.play_ep_count.append(ep_count)
        self.play_match_count.append(num_matches)
        self.play_win_rates.append(play_victories/num_matches)
        self.play_rewards_avg.append(sum(play_rewards) / num_matches)

    def log_ep_stats(self):
        if self.ep_count > 0:
            print("Episode: {}/{} | Avg. Reward: {:10.6f} | Avg. Steps: {:10.6f} | Best Reward: {}"
            .format(self.ep_count, self.ep_total, self.ep_avg_rewards[-1], self.ep_avg_steps[-1], self.best_reward), end="\r")
        else:
            print("There are no recorded episodes!")

    def log_train_stats(self):
        if self.ep_count > 0:
            print()
            print("Current Reward Avg.: " + str(sum(self.ep_rewards) / self.ep_count))
            print("Win rate: {:10.3f}%".format((self.victories / self.ep_count) * 100))
            print()
        else:
            print("There are no recorded episodes!")
    
    def plot_train_stats(self, agent):
        # Plotting average reward graph
        self.__plot_curve(range(self.ep_count), self.ep_avg_rewards, 'Episode Count',
                            'Avg. Reward', r'Reward avg. over training, $\gamma={}, \alpha={}$'.format(agent.model.gamma, agent.model.learning_rate))

        # Plotting average steps graph
        self.__plot_curve(range(self.ep_count), self.ep_avg_steps, 'Episode Count',
                            'Avg. Steps', r'Steps avg. over training, $\gamma={}, \alpha={}$'.format(agent.model.gamma, agent.model.learning_rate))
   
        if len(self.play_ep_count) > 0:
            self.__plot_bar(self.play_ep_count, [self.play_win_rates], ['Play'], 'Episode', 'Win rate (%)', 'Win rate percentage over play testing', format_percent=True)
            self.__plot_bar(self.play_ep_count, [self.play_rewards_avg], ['Play'], 'Episode', 'Reward avg.', 'Reward avg. over play testing')

    def __plot_curve(self, x, y, x_label, y_label, title):
        fig, ax = plt.subplots()
        ax.plot(x, y)

        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()

        plt.show()


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

        plt.show()


    def __lerp(self, a, b, t):
        return (1 - t) * a + t * b