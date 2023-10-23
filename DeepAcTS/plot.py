import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime
import sys

import gymnasium as gym

import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.stats as stats
import torch.distributions.multivariate_normal as mvn
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import copy
import statistics as stat

import pdb

from AI_noisy_obs_agent_AcT import ReplayMemory_AcT, Model_AcT, MVGaussianModel_AcT, Node, Agent_AcT
from dq_agent import Agent_DQN, ReplayMemory_DQN, DQN


if __name__ == '__main__':

	# Create the CartPole-v1 environment
	env = gym.make('CartPole-v1')

	# Set the number of episodes the agent will run
	num_episodes = 500

	scores = []

	# Run the episodes
	for episode in range(num_episodes):
		# Reset the environment at the beginning of each episode
		observation, _ = env.reset()
		episode_reward = 0  # Track the total reward for this episode

		# Run the episode until it is done
		terminated = False
		truncated = False
		while not (terminated or truncated):
			# Choose a random action (0 or 1 for CartPole-v1)
			action = env.action_space.sample()

			# Take the chosen action in the environment
			observation, reward, terminated, truncated, _ = env.step(action)

			# Accumulate the reward for this episode
			episode_reward += reward

		# Print the total reward achieved in this episode
		print("Episode {}: Total Reward: {}".format(episode + 1, episode_reward))

		scores.append(episode_reward)

	# Close the environment after finishing all episodes
	env.close()

	x = [i + 1 for i in range(num_episodes)]

	running_avg = np.zeros(len(scores))
	stds = np.zeros(len(scores))
	for i in range(len(running_avg)):
		running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
		stds[i] = np.std(scores[max(0, i - 100):(i + 1)])

	# Create a figure and plot the moving average
	plt.figure(figsize=(12, 6))
	# plt.plot(x, running_avg, label='Random', color='green')
	plt.plot(x, running_avg, color='green')

	# Plot the standard deviation bands
	plt.fill_between(x, running_avg - stds, running_avg + stds, color='green', alpha=0.3) #, label='Standard Deviation')

	agent_aif = Agent_AcT(sys.argv[1:])
	agent_aif.train()

	agent_DQN = Agent_DQN(sys.argv[1:])
	agent_DQN.train()

	x = [i + 1 for i in range(num_episodes)]
	# plot_learning_curve(x, agent.results, figure_file_AIF, "AcT Action Selection")

	# Calculate the moving average and standard deviation
	running_avg_AcT = np.zeros(len(agent_aif.results))
	stds_AcT = np.zeros(len(agent_aif.results))
	for i in range(len(running_avg_AcT)):
		running_avg_AcT[i] = np.mean(agent_aif.results[max(0, i - 100):(i + 1)])
		stds_AcT[i] = np.std(agent_aif.results[max(0, i - 100):(i + 1)])

	# Create a figure and plot the moving average
	# plt.figure(figsize=(12, 6))
	# plt.plot(x, running_avg_AcT, label='DAcT', color='b')
	plt.plot(x, running_avg_AcT, color='b')
	plt.fill_between(x, running_avg_AcT - stds_AcT, running_avg_AcT + stds_AcT, color='b', alpha=0.3) #, label='DAcT')

	# Calculate the moving average and standard deviation
	running_avg_DQN = np.zeros(len(agent_DQN.results))
	stds_DQN = np.zeros(len(agent_DQN.results))
	for i in range(len(running_avg_DQN)):
		running_avg_DQN[i] = np.mean(agent_DQN.results[max(0, i - 100):(i + 1)])
		stds_DQN[i] = np.std(agent_DQN.results[max(0, i - 100):(i + 1)])

	# plt.plot(x, running_avg_DQN, label='DQN', color='r')
	plt.plot(x, running_avg_DQN, color='r')
	plt.fill_between(x, running_avg_DQN - stds_DQN, running_avg_DQN + stds_DQN, color='r', alpha=0.3) #, label='DQN')

	# plt.title(f"100 step MAVG of scores")
	plt.xlabel('Episodes', fontsize = 28)
	plt.ylabel('Score', fontsize = 28)
	plt.legend()

	# Save or display the plot
	plt.show()
