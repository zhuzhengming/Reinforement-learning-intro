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
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Reinforcement learning lab 1
# group member: 2
# name: Zhengming Zhu, Xianao Lu
# ID: 19990130-2035   20021201-3338

import pickle

# Load packages
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n  # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high
# Parameters
N_episodes = 800  # Number of episodes to run for training
discount_factor = 1  # Value of gamma
max_degree = 2

R_test = False

# Reward
episode_reward_list = []  # Used to save episodes reward

# action count
steps = 0

# super parameter:
eligibility_trace = 0.1
alpha = 0.2
epsilon = 0.01
momentum = 0.2

# Linear function approximation
# we use eta =
# [0,0], [0,1], [1,0], [1,1], [2,0], [0,2], [2,1], [1,2], [2,2]
# to capture both the single variables and
# the interaction between the state variables and the action variables
eta = np.array([[i, j] for i in range(max_degree + 1) for j in range(max_degree + 1)])
# eta = eta[1:,:]

# we have one vector per action – A
omega = np.zeros((k, np.shape(eta)[0]))

if R_test is True:
# 2.e
    alphas_R = np.linspace(0.1, 0.3, 20)
    lambdas_R = np.linspace(0.0, 1.0, 20)
    n_episodes_R = 50
    rewards_alpha = np.zeros(len(alphas_R))
    rewards_lambda = np.zeros(len(lambdas_R))

# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


# Reduce the learning rate during training
def learninig_rate_decay(alpha, total_reward):
    if total_reward > -200:
        alpha *= 0.9
    elif total_reward > -130:
        alpha *= 0.78
    else:
        alpha = alpha
    return alpha


def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = np.divide((np.array(s) - np.array(low)), (np.array(high) - np.array(low)))
    return x


def fourier_basis(state):
    features = np.cos(np.pi * np.dot(eta, state))
    return features


def epsilon_greedy(epsilon, Q):
    if np.random.rand() < epsilon:
        # choose action randomly from environment
        next_action = env.action_space.sample()
    else:
        next_action = np.argmax(Q)

    return next_action


def scaling_fourier_basis(eta, alpha):
    eta_norm = np.linalg.norm(eta, 2, 1)
    for i in range(len(eta_norm)):
        if eta_norm[i] == 0:
            eta_norm[i] = 1

    scaled_alpha = np.divide(alpha, eta_norm)
    return scaled_alpha


def SGD_with_Momentum(v, delta, alpha, e, omega):
    v = momentum * v + delta * e * alpha
    # v = momentum * v + delta * np.dot(e, np.diag(alpha))  # v = mv + alpha*delta*eligibility trace
    omega += v  # omega = omega + v

    return omega

def save_pickle(omega, eta):
    W = omega
    N = eta
    data = {'W': W, 'N': N}

    filename = 'weights.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# initialization
scaled_alpha = scaling_fourier_basis(eta, alpha)

for i in range(N_episodes):
    # Reset environment data
    done = False
    initial_state = env.reset()[0]
    state = scale_state_variables(initial_state)  # initialization
    total_episode_reward = 0.
    steps = 0

    # create eligibility trace and reset
    # the same dimension as omega
    E = np.zeros((k, eta.shape[0]))
    # create term v and reset when using SGD
    Velocity = np.zeros((k, eta.shape[0]))

    while not done:
        # Take a random action
        # env.action_space.n tells you the number of actions
        # available
        # action_r = np.random.randint(0, k)
        # Q for specific state, and all actions
        Q_current = np.dot(omega, fourier_basis(state))

        # choose next action
        action = epsilon_greedy(epsilon, Q_current)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _, _ = env.step(action)
        next_state = scale_state_variables(next_state)
        steps += 1

        # Update episode reward
        total_episode_reward += reward
        # total_reward += reward

        # calculate the TD error
        Q_next = np.dot(omega, fourier_basis(next_state))

        # choose next action
        next_action = epsilon_greedy(epsilon, Q_next)  # on policy
        # next_action = np.argmax(Q_next)  # off policy

        TD_error = reward + discount_factor * Q_next[next_action] - Q_current[action]

        # update eligibility trace
        for j in range(E.shape[0]):
            if j == action:
                E[j, :] = E[j,:] * discount_factor * eligibility_trace + fourier_basis(state)
            else:
                E[j, :] *= discount_factor * eligibility_trace
        E = np.clip(E, -5, 5)  # avoid exploding gradient problem

        # update omega SGD
        omega = SGD_with_Momentum(Velocity, TD_error, scaled_alpha, E, omega)

        # update state
        state = next_state

        # explore more space when training
        if steps > 2000:
            done = True

    # update learning rate
    scaled_alpha = learninig_rate_decay(scaled_alpha, total_episode_reward)

    print("episode_num:", i)
    # Append episode reward
    episode_reward_list.append(total_episode_reward)
# rewards_lambda[q] = total_reward / n_episodes_R
# Close environment
env.close()

save_pickle(omega,eta)

if R_test is False:
    # Plot Rewards
    plt.plot([i for i in range(1, N_episodes + 1)], episode_reward_list, label='Episode reward')
    plt.plot([i for i in range(1, N_episodes + 1)], running_average(episode_reward_list, 10),
             label='Average episode reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


#     # Plot optimal val func
#     x = np.linspace(0,1,100)
#     y = np.linspace(0,1,100)
#     X, Y = np.meshgrid(x, y)
#     V = np.array([[max(np.dot(omega, fourier_basis(np.array([i,j])))) for i in x] for j in y])
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(x, y, V, cmap='viridis')
#
#     ax.set_xlabel('position')
#     ax.set_ylabel('velocity')
#     ax.set_zlabel('value function')
#
#     plt.show()
#
#     # Plot optimal policy
#     x = np.linspace(0,1,100)
#     y = np.linspace(0,1,100)
#     A = np.array([[np.argmax(np.dot(omega, fourier_basis(np.array([i,j])))) for i in x] for j in y])
#
#     plt.figure(figsize=(8, 6))
#     plt.imshow(A, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
#
#     plt.colorbar(label='optimal policy')
#     plt.xlabel('position')
#     plt.ylabel('velocity')
#     plt.show()
#
# else:
#     plt.figure(figsize=(6, 6))
#     plt.plot(lambdas_R, rewards_lambda, marker='o')
#     plt.xlabel('Eligibility Trace (λ)')
#     plt.ylabel('Average Total Reward')
#     plt.title('Average Total Reward vs Eligibility Trace')
#     plt.show()