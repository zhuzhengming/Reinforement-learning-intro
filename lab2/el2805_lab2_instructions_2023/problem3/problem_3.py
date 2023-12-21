import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import trange
from PPO_agent import RandomAgent, PPO_Agent
import torch


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# import running environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

dim_actions = len(env.action_space.high)  # dimensionality of the action
dim_states = len(env.observation_space.high)  # dimensionality of the state

# macro variables
MODE = ['TRAIN', 'GAMMA_COMP', 'EPSILON_COMP', 'POLICY_COMP', 'POLICY_SHOW']
N_episodes = 1600               # Number of episodes to run for training
n_ep_running_average = 50      # Running average of 50 episodes

def draw_plot(episode_reward_list, episode_number_of_steps):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, N_episodes + 1)], episode_reward_list, label='Episode reward', alpha=0.3)
    ax[0].plot([i for i in range(1, N_episodes + 1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, N_episodes + 1)], episode_number_of_steps, label='Steps per episode', alpha=0.3)
    ax[1].plot([i for i in range(1, N_episodes + 1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()


def train_process(gamma, epsilon, alpha_critic, alpha_actor, epoch_num, episode_num, max_step, buffer_size, SAVE_NET = True):
    # trainning parameter setting

    PPO = PPO_Agent(gamma, epsilon, alpha_critic, alpha_actor, epoch_num, episode_num, max_step, dim_states, dim_actions, buffer_size)

    episode_reward_list = []  # Used to save episodes reward
    episode_number_of_steps = []
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    for i in EPISODES:
        state = env.reset()[0]
        PPO.buffer_clear()
        done = False
        t = 0
        total_episode_reward = 0
        while not done:
            action = PPO.action_gen(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            PPO.buffer_push((state, action, reward, next_state, done))

            total_episode_reward += reward
            state = next_state
            if t >= 1000:
                break
            t += 1

        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                i, total_episode_reward, t,
                running_average(episode_reward_list, n_ep_running_average)[-1],
                running_average(episode_number_of_steps, n_ep_running_average)[-1]))
        PPO.update()
    if(SAVE_NET):
        PPO.save_network()
    draw_plot(episode_reward_list, episode_number_of_steps)

def gamma_comp(gamma_0, gamma_1):
    # trainning parameter setting
    epsilon = 0.2
    alpha_critic = 1e-3            # learning rate for critic
    alpha_actor = 1e-5             # learning rate for actor
    epoch_num = 10
    episode_num = N_episodes
    max_step = 1000
    buffer_size = 20000

    gamma = gamma_0            # discount factor
    train_process(gamma, epsilon, alpha_critic, alpha_actor, epoch_num, episode_num, max_step, buffer_size, SAVE_NET=False)

    gamma = gamma_1
    train_process(gamma, epsilon, alpha_critic, alpha_actor, epoch_num, episode_num, max_step, buffer_size, SAVE_NET=False)

def epsilon_comp(epsilon_0, epsilon_1):
    # trainning parameter setting
    gamma = 0.99
    alpha_critic = 1e-3            # learning rate for critic
    alpha_actor = 1e-5             # learning rate for actor
    epoch_num = 10
    episode_num = N_episodes
    max_step = 1000
    buffer_size = 20000

    epsilon = epsilon_0
    train_process(gamma, epsilon, alpha_critic, alpha_actor, epoch_num, episode_num, max_step, buffer_size, SAVE_NET=False)

    epsilon = epsilon_1
    train_process(gamma, epsilon, alpha_critic, alpha_actor, epoch_num, episode_num, max_step, buffer_size, SAVE_NET=False)

def Policy_Show():
    try:
        model_critic = torch.load('neural-network-3-critic.pth', map_location=torch.device('cpu'))
        model_actor = torch.load('neural-network-3-actor.pth', map_location=torch.device('cpu'))
    except:
        print('File neural-network.pth not found!')
        exit(-1)

    y_list = np.linspace(0, 1.5, 100)
    omega_list = np.linspace(-np.pi, np.pi, 100)

    Y, Omega = np.meshgrid(y_list, omega_list)
    V_value_list = np.zeros(Y.shape)
    mean_action = np.zeros(Y.shape)

    for i in range(len(y_list)):
        for j in range(len(omega_list)):
            state = torch.tensor([0, y_list[i], 0, 0, omega_list[j], 0, 0, 0], dtype=torch.float32)

            with torch.no_grad():
                mean, var = model_actor(state)
                V_value = model_critic(state).item()

            mean_action[j, i] = mean[1].item()
            V_value_list[j, i] = V_value

    figure, axes = plt.subplots(subplot_kw={"projection": "3d"})
    plot = axes.plot_surface(Omega, Y, mean_action, cmap="coolwarm")
    axes.set_xlabel(r"$\omega$")
    axes.set_ylabel(r"$y$")
    axes.set_zlabel(r"$\mu_{\theta,2}(s)$")
    axes.set_title('V-value for Varying Height and Angle')
    figure.colorbar(plot, location="left")
    figure.show()

    figure, axes = plt.subplots(subplot_kw={"projection": "3d"})
    plot = axes.plot_surface(Omega, Y, V_value_list, cmap="coolwarm")
    axes.set_xlabel(r"$\omega$")
    axes.set_ylabel(r"$y$")
    axes.set_zlabel(r"$V_{\omega}(s)$")
    axes.set_title('Mean for engine direction')
    figure.colorbar(plot, location="left")
    figure.show()

def Policy_Comp():
    try:
        model = torch.load('neural-network-3-actor.pth')
    except:
        print('File neural-network.pth not found!')
        exit(-1)

    N_EPISODES = 50
    EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
    # test trained optimal policy
    episode_reward_list_optim = []  # Used to store episodes reward
    for i in EPISODES:
        EPISODES.set_description("Episode {}".format(i))
        # Reset enviroment data
        done = False
        state = env.reset()[0]
        total_episode_reward = 0.
        while not done:
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            mu, var = model(torch.tensor([state], device='cuda'))
            mu = mu.cpu()
            mu = mu.detach().numpy()
            var = var.cpu()
            std = torch.sqrt(var).detach().numpy()
            actions = np.clip(np.random.normal(mu, std), -1, 1).flatten()
            next_state, reward, done, truncated, _ = env.step(actions)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
        # Append episode reward
        episode_reward_list_optim.append(total_episode_reward)
        # Close environment
        env.close()

    # test random policy
    episode_reward_list_random = []  # Used to store episodes reward
    agent = RandomAgent(dim_actions)
    EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
    for i in EPISODES:
        # Reset enviroment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        while not done:
            # Take a random action
            action = agent.forward(state)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, truncated, _ = env.step(action)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t+= 1
        # Append episode reward
        episode_reward_list_random.append(total_episode_reward)
        # Close environment
        env.close()

    # plot rewards got from two agent
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    ax.plot([i for i in range(1, len(episode_reward_list_optim) + 1)], episode_reward_list_optim,
               label='PPO agent Episode reward')
    ax.plot([i for i in range(1, len(episode_reward_list_random) + 1)], episode_reward_list_random,
            label='Random agent Episode reward')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Total reward')
    ax.set_title('Total Reward Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.show()


if __name__ == "__main__":
    mode = 'POLICY_SHOW'

    if mode not in MODE:
        error = 'ERROR: the argument method must be in {}'.format(MODE)
        raise NameError(error)

    if mode == 'TRAIN':
        print('Train mode')
        gamma = 0.99  # discount factor
        epsilon = 0.1
        alpha_critic = 1e-3  # learning rate for critic
        alpha_actor = 1e-5  # learning rate for actor
        epoch_num = 10
        episode_num = N_episodes
        max_step = 1000
        buffer_size = 20000
        train_process(gamma, epsilon, alpha_critic, alpha_actor, epoch_num, episode_num, max_step, buffer_size)

    if mode == 'GAMMA_COMP':
        print("Compare Gamma")
        gamma_0 = 0.1
        gamma_1 = 1
        gamma_comp(gamma_0, gamma_1)

    if mode == 'EPSILON_COMP':
        print("Compare Epsilon")
        epsilon_0 = 0.05
        epsilon_1 = 0.5
        epsilon_comp(epsilon_0, epsilon_1)

    if mode == 'POLICY_SHOW':
        print("Policy Show")
        Policy_Show()

    if mode == 'POLICY_COMP':
        print("Policy Comparison")
        Policy_Comp()

