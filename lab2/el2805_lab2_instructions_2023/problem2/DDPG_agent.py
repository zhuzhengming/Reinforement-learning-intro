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
# Last update: 26th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
from model import Actor_MLP, Critic_MLP, Experience
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

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

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)


class DDPGAgent:
    def __init__(self,
                 state_size, action_size,
                 batch_size=64, gamma=0.95,
                 actor_lr=5e-5, critic_lr=5e-4,
                 device='cpu'):

        # initialize parameters
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = device

        # create network for actor and critic
        # actor
        self.actor_main_net = Actor_MLP(state_size, action_size, device)
        self.actor_target_net = Actor_MLP(state_size, action_size, device)
        self.actor_target_net.eval()
        self.actor_optimizer = optim.Adam(self.actor_main_net.parameters(), lr=self.actor_lr)

        # critic
        self.critic_main_net = Critic_MLP(state_size, action_size, device)
        self.critic_target_net = Critic_MLP(state_size, action_size, device)
        self.critic_target_net.eval()
        self.critic_optimizer = optim.Adam(self.critic_main_net.parameters(), lr=self.critic_lr)

    # add a noise signal
    def noisy_action(self, mu, sigma, n, action):
        omega = np.random.normal(0, sigma, self.action_size)
        n = -mu * n + omega

        action = action + n
        action = np.clip(action, -1, 1).reshape(-1)
        return action, n

    def critic_train(self, buffer):
        if len(buffer) < self.batch_size:
            return

        # sampling and unpack for training
        transitions = buffer.sample(self.batch_size)
        # unpack transition and pack into batch
        batch = Experience(*zip(*transitions))

        states = torch.FloatTensor(batch.state).to(self.device)
        actions = torch.FloatTensor(batch.action).to(self.device)
        next_state = torch.FloatTensor(batch.next_state).to(self.device)
        reward = torch.FloatTensor(batch.reward).to(self.device)
        done = torch.FloatTensor(batch.done).to(self.device)

        # set gradient as o
        self.critic_optimizer.zero_grad()

        # because the output of network is all action value.
        Q_current = self.critic_main_net(states, actions).squeeze()

        # calculate target Q value
        with torch.no_grad():
            next_action = self.actor_target_net(next_state)
            target_Q = self.critic_target_net(next_state, next_action).squeeze()
            # if done just current reward
            y = reward + (1 - done) * self.gamma * target_Q

        # define the MSE loss
        critic_loss = F.mse_loss(Q_current, y)

        # calculate gradient
        critic_loss.backward()

        # clip the gradient
        nn.utils.clip_grad_norm_(self.critic_main_net.parameters(), 1.0)

        # start backward pass
        self.critic_optimizer.step()

    def actor_train(self, buffer):
        if len(buffer) < self.batch_size:
            return

            # sampling and unpack for training
        transitions = buffer.sample(self.batch_size)
        # unpack transition and pack into batch
        batch = Experience(*zip(*transitions))

        states = torch.FloatTensor(batch.state).to(self.device)

        # set gradient as o
        self.actor_optimizer.zero_grad()

        # loss
        actor_loss = -self.critic_main_net(states, self.actor_main_net(states)).squeeze().mean()

        # calculate gradient
        actor_loss.backward()

        # clip the gradient
        nn.utils.clip_grad_norm_(self.actor_main_net.parameters(), 1.0)

        # start backward pass
        self.actor_optimizer.step()

    def save_network(self):
        torch.save(self.actor_main_net, 'neural-network-2-actor.pth')
        print("neural-network-2-actor saved!")
        torch.save(self.critic_main_net, 'neural-network-2-critic.pth')
        print("neural-network-2-critic saved!")

