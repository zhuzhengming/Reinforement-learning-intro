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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 29th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim

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
                    the parent class Agent
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)


Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class Critic_Network(nn.Module):
    def __init__(self, dev, input_size, output_size):
        super().__init__()

        neuron_num_l1 = 400
        neuron_num_l2 = 200

        self.input_layer = nn.Linear(input_size, neuron_num_l1, device=dev)
        self.input_layer_activation = nn.ReLU()

        self.hidden_layer = nn.Linear(neuron_num_l1, neuron_num_l2, device=dev)
        self.hidden_layer_activation = nn.ReLU()

        self.output_layer = nn.Linear(neuron_num_l2, output_size, device=dev)

    def forward(self, x):
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        l2 = self.hidden_layer(l1)
        l2 = self.hidden_layer_activation(l2)

        out = self.output_layer(l2)

        return out


class Actor_Network(nn.Module):
    def __init__(self, dev, input_size, output_size):
        super().__init__()

        neuron_num_input = 400
        neuron_num_hiden = 200

        self.input_layer = nn.Linear(input_size, neuron_num_input, device=dev)
        self.input_layer_activation = nn.ReLU()

        self.hidden_layer_mean = nn.Linear(neuron_num_input, neuron_num_hiden, device=dev)
        self.hidden_layer_mean_activation = nn.ReLU()
        self.output_layer_mean = nn.Linear(neuron_num_hiden, output_size, device=dev)
        self.output_layer_mean_activation = nn.Tanh()

        self.hidden_layer_var = nn.Linear(neuron_num_input, neuron_num_hiden, device=dev)
        self.hidden_layer_var_activation = nn.ReLU()
        self.output_layer_var = nn.Linear(neuron_num_hiden, output_size, device=dev)
        self.output_layer_var_activation = nn.Sigmoid()

    def forward(self,x):

        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        l2_mean = self.hidden_layer_mean(l1)
        l2_mean = self.hidden_layer_mean_activation(l2_mean)
        out_mean = self.output_layer_mean(l2_mean)
        out_mean = self.output_layer_mean_activation(out_mean)

        l2_var = self.hidden_layer_var(l1)
        l2_var = self.hidden_layer_var_activation(l2_var)
        out_var = self.output_layer_var(l2_var)
        out_var = self.output_layer_var_activation(out_var)

        return out_mean, out_var


class PPO_Agent(object):
    def __init__(self, gamma, epsilon, alpha_critic, alpha_actor, epoch_num, episode_num, max_step, state_dim, action_dim, buffer_size):
        self.gamma = gamma                          #discount factor
        self.epsilon = epsilon
        self.alpha_critic = alpha_critic            # learning rate for critic network
        self.alpha_actor = alpha_actor              # learning rate for critic network
        self.epoch_num = epoch_num
        self.episode_num = episode_num
        self.max_step = max_step
        self.buffer_size = buffer_size

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Trainning on", self.dev)
        self.critic = Critic_Network(self.dev, state_dim, 1)
        self.actor = Actor_Network(self.dev, state_dim, action_dim)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.alpha_critic)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.alpha_actor)

        self.buffer = deque([], maxlen=self.buffer_size)

    def buffer_clear(self):
        self.buffer.clear()

    def buffer_push(self, experience):
        # record an experience
        self.buffer.append(experience)

    def action_gen(self, state):
        mean, var = self.actor(torch.tensor(state, device= self.dev))
        mean_cpu = mean.cpu()
        mean_cpu = mean_cpu.detach().numpy()
        var_cpu = var.cpu()
        var_cpu = np.sqrt(var_cpu.detach().numpy())
        action_0 = np.random.normal(mean_cpu[0], var_cpu[0])
        action_1 = np.random.normal(mean_cpu[1], var_cpu[1])
        action = np.clip([action_0, action_1], -1, 1)
        return action

    def Gaussian_Prob_Calc(self, mean, var, action):
        distr = torch.distributions.Normal(mean, torch.sqrt(var))
        log_prob = distr.log_prob(action)
        prob_sep = torch.exp(log_prob)
        prob = prob_sep[:, 0] * prob_sep[:, 1]
        return prob

    def update(self):

        # batch = Experience(*zip(*self.buffer))
        #
        # state = batch.state
        # action = batch.action
        # reward = batch.reward
        # next_state = batch.next_state
        # done = batch.done

        state, action, reward, next_state, done = zip(*self.buffer)
        buffer_len = len(state)

        #compute G_i
        G = np.zeros(buffer_len)
        G[-1] = reward[-1]
        for i in reversed(range(buffer_len-1)):
            G[i] = G[i+1] * self.gamma + reward[i]
        G = torch.tensor(G, requires_grad=True,dtype=torch.float32, device=self.dev)

        tensor_state = torch.tensor(state, requires_grad=True,dtype=torch.float32, device=self.dev)
        tensor_action = torch.tensor(action, requires_grad=True,dtype=torch.float32, device=self.dev)


        mean, var = self.actor(tensor_state)
        old_probs = self.Gaussian_Prob_Calc(mean, var, tensor_action).detach()

        for i in range(self.epoch_num):
            # Critic Network update
            self.critic_optim.zero_grad()
            V_w = self.critic(tensor_state).squeeze()
            loss = nn.functional.mse_loss(G, V_w)
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
            self.critic_optim.step()

            # Actor Network update
            V_w = self.critic(torch.tensor(state, dtype=torch.float32, device=self.dev)).squeeze()
            Fai = G - V_w           # advantage
            mean_t, var_t = self.actor(tensor_state)
            probs = self.Gaussian_Prob_Calc(mean_t, var_t, tensor_action)
            r_theta = probs / old_probs
            value_1 = r_theta * Fai
            c_epsilon = torch.clamp(r_theta, min=(1-self.epsilon), max=(1+self.epsilon))
            value_2 = c_epsilon * Fai
            loss_actor = torch.min(value_1, value_2)
            loss_actor = -torch.mean(loss_actor)
            self.actor_optim.zero_grad()
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            self.actor_optim.step()

    def save_network(self):
        torch.save(self.actor, 'neural-network-3-actor.pth')
        print("neural-network-2-actor saved!")
        torch.save(self.critic, 'neural-network-3-critic.pth')
        print("neural-network-2-critic saved!")