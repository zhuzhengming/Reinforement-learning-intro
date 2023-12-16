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
import torch


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
MODE = ['Train', 'Gamma_test', 'Episodes_test', 'Memory_size_test', 'Restriction', 'Comparison']
PLOT = True
SAVE_NET = True
STOP = True

# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v2')
env.reset()

# Default Parameters
N_episodes = 400  # Number of episodes
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
hidden_dim = 32


def Train_mode(gamma, Memory_size, episode_num):
    ########################################################################################
    # TRAIN mode
    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []  # this list contains the total reward per episode
    episode_number_of_steps = []  # this list contains the number of steps per episode

    # create DQN agent, as well as network
    agent = DQNAgent(dim_state, n_actions, batch_size,
                     gamma, lr, hidden_dim)

    # init experience replay buffer
    buffer = ExperienceReplayBuffer(Memory_size)
    # fill with random experiences
    buffer.fill_random_experience()

    # Training process

    # trange is an alternative to range in python, from the tqdm library
    # It shows a nice progression bar that you can update with useful information
    EPISODES = trange(episode_num, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset environment data and initialize variables
        done = False
        state = env.reset()[0]  # Initialize environment and read initial state s0
        total_episode_reward = 0.
        t = 0  # t ← 0

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
        if STOP:
            if average_episode_reward > 70:
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

        ax[1].plot([i for i in range(1, len(episode_number_of_steps) + 1)], episode_number_of_steps,
                   label='Steps per episode')
        ax[1].plot([i for i in range(1, len(episode_number_of_steps) + 1)], running_average(
            episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('Total number of steps')
        ax[1].set_title('Total number of steps vs Episodes')
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        plt.show()


def Comparison(episode_num):
    # Load model
    try:
        model = torch.load('neural-network-1.pth')
    except:
        print('File neural-network-1.pth not found!')
        exit(-1)

    DQN_episode_reward_list = []
    R_episode_reward_list = []

    agentRandom = RandomAgent(n_actions)

    EPISODES = trange(episode_num, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset environment data and initialize variables
        DQN_done = False
        DQN_state = env.reset()[0]  # Initialize environment and read initial state s0
        DQN_total_episode_reward = 0.
        while not DQN_done:
            # choose action
            with torch.no_grad():
                q_values = model(torch.tensor([DQN_state]))
            _, DQN_actions = torch.max(q_values, axis=1)
            DQN_next_state, DQN_reward, DQN_done, _, _ = env.step(DQN_actions.item())

            # update episode reward
            DQN_total_episode_reward += DQN_reward

            # update the state
            DQN_state = DQN_next_state

        # Append episode reward
        DQN_episode_reward_list.append(DQN_total_episode_reward)

        R_done = False
        R_state = env.reset()[0]  # Initialize environment and read initial state s0
        R_total_episode_reward = 0.

        while not R_done:
            # choose action
            R_action = agentRandom.forward(R_state)
            R_next_state, R_reward, R_done, _, _ = env.step(R_action)

            # update episode reward
            R_total_episode_reward += R_reward

            # update the state
            R_state = R_next_state

        # Append episode reward
        R_episode_reward_list.append(R_total_episode_reward)

        env.close()

        EPISODES.set_description(
            "Episode {}".format(i))

    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    ax[0].plot([i for i in range(1, len(DQN_episode_reward_list) + 1)], DQN_episode_reward_list,
               label='DQN Episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('DQN Total Reward')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, len(R_episode_reward_list) + 1)], R_episode_reward_list,
               label='Random Episode reward')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total reward')
    ax[1].set_title('Random Total Reward')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()


def Restriction_test():
    # Load model
    try:
        model = torch.load('neural-network-1.pth')
    except:
        print('File neural-network-1.pth not found!')
        exit(-1)

    y_list = np.linspace(0, 1.5, 100)
    omega_list = np.linspace(-np.pi, np.pi, 100)

    Y, Omega = np.meshgrid(y_list, omega_list)
    Max_Q_values = np.zeros(Y.shape)
    Optimal_action = np.zeros(Y.shape)

    for i in range(len(y_list)):
        for j in range(len(omega_list)):
            state = torch.tensor((0, y_list[i], 0, 0, omega_list[j], 0, 0, 0), dtype=torch.float32)

            with torch.no_grad():
                q_value = model(state)
                max_q_value = torch.max(q_value).item()
                optimal_action = torch.argmax(q_value).item()

            # Record
            Optimal_action[j, i] = optimal_action
            Max_Q_values[j, i] = max_q_value

    # plot
    fig1 = plt.figure(figsize=(10, 6))
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot_surface(Omega, Y, Max_Q_values, cmap='viridis')

    ax.set_xlabel('Angle ω')
    ax.set_ylabel('Height y')
    ax.set_zlabel('Max Q-value')
    ax.set_title('Max Q-value for Varying Height and Angle')

    plt.figure(figsize=(10, 6))
    plt.imshow(Optimal_action, extent=[0, 1.5, -np.pi, np.pi], origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Optimal Action')
    plt.xlabel('Height y')
    plt.ylabel('Angle ω')
    plt.title('Optimal Action Heatmap for Varying Height and Angle')

    plt.show()


if __name__ == "__main__":
    # initialize mode
    mode = 'Restriction'

    if mode not in MODE:
        error = 'ERROR: the argument method must be in {}'.format(MODE)
        raise NameError(error)

    if mode == 'Train':
        print('Train mode')
        Train_mode(discount_factor, capacity, N_episodes)

    elif mode == 'Gamma_test':
        print('Gamma_test mode')
        SAVE_NET = False
        gamma_0 = 0.8
        gamma_1 = 1
        Train_mode(gamma_1, capacity, N_episodes)

    elif mode == 'Episodes_test':
        print('Episodes_test mode')
        SAVE_NET = False
        STOP = False
        Episodes_0 = 600
        Train_mode(discount_factor, capacity, Episodes_0)

    elif mode == 'Memory_size_test':
        print('Memory_size_test mode')
        SAVE_NET = False
        Memory_size_1 = 20000
        Train_mode(discount_factor, Memory_size_1, N_episodes)

    elif mode == 'Restriction':
        print('Restriction mode')
        Restriction_test()

    elif mode == 'Comparison':
        print('Comparison mode')
        episodes_num = 50
        Comparison(episodes_num)
