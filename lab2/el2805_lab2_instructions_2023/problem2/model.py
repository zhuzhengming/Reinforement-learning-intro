# Experience replay buffer and MLP network class
import torch
from collections import namedtuple, deque
import random
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ExperienceReplayBuffer(object):
    def __init__(self, capacity):
        # initialize a buffer with specific length
        # FIFO
        self.buffer = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, experience):
        # record an experience
        self.buffer.append(experience)

    def sample(self, batch_size):
        # sample a batch of experience
        if batch_size <= len(self.buffer):
            return random.sample(self.buffer, batch_size)
        else:
            raise IndexError('no enough experience for sampling!')

    def fill_random_experience(self):
        env = gym.make('LunarLanderContinuous-v2')
        env.reset()
        while len(self.buffer) < self.capacity:
            state = env.reset()[0]
            done = False
            while not done:
                # generate random experience
                action = np.clip(-1 + 2 * np.random.rand(2), -1, 1)
                next_state, reward, done, _, _ = env.step(action)
                experience = (state, action, next_state, reward, done)
                self.push(experience)
        env.close()

    def __len__(self):
        return len(self.buffer)


# Actor MLP network
# hidden layers: 3 layer
# input: state
# output: optimal action vector
class Actor_MLP(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()

        hidden_dim_l1 = 400
        hidden_dim_l2 = 200

        # layer 1
        self.layer1 = nn.Linear(state_dim, hidden_dim_l1, device=device)
        # layer 2
        self.layer2 = nn.Linear(hidden_dim_l1, hidden_dim_l2, device=device)
        # output layer
        self.output = nn.Linear(hidden_dim_l2, action_dim, device=device)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        action = F.tanh(self.output(x))  # constraint the action to be between [−1, 1]
        return action


# Critic MLP network
# hidden layers: 3 layer
# input: state
# output: single value
class Critic_MLP(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()

        hidden_dim_l1 = 400
        hidden_dim_l2 = 200

        # layer 1
        self.layer1 = nn.Linear(state_dim, hidden_dim_l1, device=device)
        # layer 2: concatenate the action
        self.layer2 = nn.Linear(hidden_dim_l1 + action_dim, hidden_dim_l2, device=device)
        # output layer: evaluate single (state, action) value
        self.output = nn.Linear(hidden_dim_l2, 1, device=device)

    def forward(self, state, action):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(torch.cat([x, action], dim=1)))
        value = self.output(x)
        return value
