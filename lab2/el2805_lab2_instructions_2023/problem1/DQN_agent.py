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
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
from model import MLP, Experience
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


# DQN agent
class DQNAgent:
    def __init__(self,
                 state_size, action_size,
                 batch_size=64, gamma=0.95,
                 lr=0.001, hidden_dim=64):

        # initialize parameters
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr

        # input: state
        # output: Q value for every action
        self.main_net = MLP(state_size, hidden_dim, action_size)
        # update target_net for C episodes
        self.target_net = MLP(state_size, hidden_dim, action_size)
        self.target_net.eval()

        # optimizer: only for main_net
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=self.lr)

    def epsilon_greedy(self, state, epsilon):
        if np.random.uniform(0,1) < epsilon:
            action = np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                Q_value = self.main_net(torch.FloatTensor(state))
                action = Q_value.max(0)[1].item()

        return action

    def epsilon_decay(self, epsilon_min, epsilon_max, k, Z):
        # way 1
        # epsilon_k = max(epsilon_min,
        #                 epsilon_max - (epsilon_max - epsilon_min)*(k - 1)/(Z - 1))
        # way 2
        epsilon_k = max(epsilon_min, epsilon_max * (epsilon_min/epsilon_max)**((k-1)/(Z-1)))
        return epsilon_k

    def train(self, buffer):
        if len(buffer) < self.batch_size:
            return

        # sampling and unpack for training
        transitions = buffer.sample(self.batch_size)
        # unpack transition and pack into batch
        batch = Experience(*zip(*transitions))

        states = torch.FloatTensor(batch.state)
        actions = torch.FloatTensor(batch.action)
        next_state = torch.FloatTensor(batch.next_state)
        reward = torch.FloatTensor(batch.reward)
        done = torch.FloatTensor(batch.done)

        # calculate current Q value
        # size: 1*batch_size
        # because the output of network is all action value.
        Q_current = self.main_net(states).gather(1, torch.tensor(actions, dtype=torch.int64)
                                                 .unsqueeze(1)).squeeze(1)

        # calculate max target Q value
        # find max Q value for every row
        target_Q = self.target_net(next_state).max(1)[0]
        # if done just current reward
        y = reward + (1 - done) * self.gamma * target_Q

        # define the MSE loss
        loss = F.mse_loss(Q_current, y)

        # calculate gradient
        loss.backward()

        # clip the gradient
        nn.utils.clip_grad_norm_(self.main_net.parameters(), 1.0)

        # start backward pass
        self.optimizer.step()

    def save_network(self):
        torch.save(self.main_net, 'neural-network-1.pth')
        print("neural network is saved!")
