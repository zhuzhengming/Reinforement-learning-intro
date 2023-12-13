# Experience replay buffer and MLP network class

from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

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
        env = gym.make('LunarLander-v2')
        env.reset()
        while len(self.buffer) < self.capacity:
            state = env.reset()[0]
            done = False
            while not done:
                # generate random experience
                action = random.randint(0, env.action_space.n - 1)
                next_state, reward, done, _, _ = env.step(action)
                experience = (state, action, next_state, reward, done)
                self.push(experience)
        env.close()

    def __len__(self):
        return len(self.buffer)


# MLP network
# hidden layers: 2 layer
# neurons for every layer: 64
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # layer 1
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        # layer 2
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        # output layer
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)  # no activate function
        return x
