# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent, DQNAgent
from model import ExperienceReplayBuffer


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


# debug
PLOT = True
SAVE_NET = True

# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 600  # Number of episodes
discount_factor = 0.98  # Value of the discount factor
n_ep_running_average = 50  # Running average of 50 episodes
n_actions = env.action_space.n  # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
lr = 0.0005  # learning rate

# Decay ε
epsilon_min = 0.05
epsilon_max = 0.99
Z = 0.92 * N_episodes

batch_size = 64  # batch size for experience sampling
capacity = 10000  # capacity of experience buffer
C = int(capacity / batch_size)  # update target network after C steps
hidden_dim = 64

########################################################################################
# TRAIN mode
# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []  # this list contains the total reward per episode
episode_number_of_steps = []  # this list contains the number of steps per episode

# create DQN agent, as well as network
agent = DQNAgent(dim_state, n_actions, batch_size,
                 discount_factor, lr, hidden_dim)

# init experience replay buffer
buffer = ExperienceReplayBuffer(capacity)
# fill with random experiences
buffer.fill_random_experience()

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset environment data and initialize variables
    done = False
    state = env.reset()[0]  # Initialize environment and read initial state s0
    total_episode_reward = 0.
    t = 0  # t ← 0
    step_num = 1

    # update epsilon
    epsilon_k = agent.epsilon_decay(epsilon_min, epsilon_max, i, Z)

    while not done:
        # Take εk-greedy action at (with respect to Qθ(a, st))
        action = agent.epsilon_greedy(state, epsilon_k)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _, _ = env.step(action)

        # add experience into buffer
        experience = (state, action, next_state, reward, done)
        buffer.push(experience)

        # sample experience and train
        agent.train(buffer)

        # update target network after C steps
        if i % C == 0:
            agent.target_net.load_state_dict(agent.main_net.state_dict())

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t += 1
        step_num += 1

        # a limit of 1000 steps because of infinite fuel
        if t > 1000:
            break

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()
    average_episode_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
    average_episode_step = running_average(episode_number_of_steps, n_ep_running_average)[-1]
    if average_episode_reward > n_ep_running_average:
        break
    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            average_episode_reward,
            average_episode_step))

if SAVE_NET:
    agent.save_network()

if PLOT:
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    ax[0].plot([i for i in range(1, len(episode_reward_list) + 1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, len(episode_reward_list) + 1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, len(episode_number_of_steps) + 1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, len(episode_number_of_steps) + 1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()
