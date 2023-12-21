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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent, DDPGAgent
from model import ExperienceReplayBuffer
from DDPG_soft_updates import soft_updates


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


# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# debug
MODE = ['Train', 'Gamma_test', 'Episodes_test', 'Memory_size_test', 'Restriction', 'Comparison']
PLOT = True
SAVE_NET = True

# Parameters
N_episodes = 300  # Number of episodes to run for training
discount_factor = 0.98  # Value of gamma
n_ep_running_average = 50  # Running average of 50 episodes
dim_actions = len(env.action_space.high)  # dimensionality of the action
dim_states = len(env.observation_space.high)  # dimensionality of the state
actor_lr = 5e-5  # learning rate
critic_lr = 5e-4
epsilon = 1e-3
batch_size = 64  # batch size for experience sampling
capacity = 30000  # capacity of experience buffer
tau = 0.0015
d = 2

# noise
mu = 0.15
sigma = 0.2


def Train_mode(gamma, Memory_size, episode_num):
    # use GPU for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Reward
    episode_reward_list = []  # Used to save episodes reward
    episode_number_of_steps = []

    # Agent initialization
    agent = DDPGAgent(dim_states, dim_actions, batch_size,
                      gamma, actor_lr, critic_lr, device)

    # init experience replay buffer
    buffer = ExperienceReplayBuffer(Memory_size)
    # fill with random experiences
    buffer.fill_random_experience()

    # Training process
    EPISODES = trange(episode_num, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset environment data
        done = False
        state = env.reset()[0]
        total_episode_reward = 0.
        t = 0
        n = np.array([0, 0])
        while not done:

            # Take action
            action = agent.actor_main_net(torch.tensor(state, device=device)).detach().cpu().numpy()
            action, n = agent.noisy_action(mu, sigma, n, action)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, terminate, truncated, _ = env.step(action)
            done = terminate or truncated
            # done = terminate

            # add experience into buffer
            experience = (state, action, next_state, reward, done)
            buffer.push(experience)

            # sample experience and train
            agent.critic_train(buffer)

            if t % d == 0:
                # sample experience and train
                agent.actor_train(buffer)
                # soft update
                agent.actor_target_net = soft_updates(agent.actor_main_net,
                                                      agent.actor_target_net, tau)
                agent.critic_target_net = soft_updates(agent.critic_main_net,
                                                       agent.critic_target_net, tau)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1

            # # a limit of 1000 steps because of infinite fuel
            # if t > 1000:
            #     break

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()
        average_episode_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
        average_episode_step = running_average(episode_number_of_steps, n_ep_running_average)[-1]
        if average_episode_reward > 200:
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
        print("save the network")

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


def Restriction_test():
    # Load model
    try:
        model_critic = torch.load('neural-network-2-critic.pth', map_location=torch.device('cpu'))
        model_actor = torch.load('neural-network-2-actor.pth', map_location=torch.device('cpu'))
    except:
        print('File neural-network.pth not found!')
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
                # choose action by actor
                optimal_action = model_actor(state)
                # evaluate value by critic
                q_value = model_critic(state.reshape((1, -1)), optimal_action.reshape((1, -1))).item()

            # Record
            Optimal_action[j, i] = optimal_action[1].item()
            Max_Q_values[j, i] = q_value

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


def Comparison(episode_num):
    # Load model
    try:
        model_actor = torch.load('neural-network-2-actor.pth', map_location=torch.device('cpu'))
    except:
        print('File neural-network.pth not found!')
        exit(-1)

    DDPG_episode_reward_list = []
    R_episode_reward_list = []

    agentRandom = RandomAgent(dim_actions)

    EPISODES = trange(episode_num, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset environment data and initialize variables
        DDPG_done = False
        DDPG_state = env.reset()[0]  # Initialize environment and read initial state s0
        DDPG_total_episode_reward = 0.
        while not DDPG_done:
            with torch.no_grad():
                # Take action
                DDPG_action = model_actor(torch.FloatTensor(DDPG_state)).cpu().detach().numpy()

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            DDPG_next_state, DDPG_reward, DDPG_done, _, _ = env.step(DDPG_action)

            # update episode reward
            DDPG_total_episode_reward += DDPG_reward

            # update the state
            DDPG_state = DDPG_next_state

        # Append episode reward
        DDPG_episode_reward_list.append(DDPG_total_episode_reward)

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
    ax[0].plot([i for i in range(1, len(DDPG_episode_reward_list) + 1)], DDPG_episode_reward_list,
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

if __name__ == "__main__":
    # initialize mode
    mode = 'Train'

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

    elif mode == 'Memory_size_test':
        print('Memory_size_test mode')
        SAVE_NET = False
        Memory_size_1 = 20000
        Memory_size_2 = 40000
        Train_mode(discount_factor, Memory_size_2, N_episodes)

    elif mode == 'Restriction':
        print('Restriction mode')
        Restriction_test()

    elif mode == 'Comparison':
        print('Comparison mode')
        episodes_num = 50
        Comparison(episodes_num)
