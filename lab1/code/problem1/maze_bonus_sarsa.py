# Reinforcement learning lab 1
# group member: 2
# name: Zhengming Zhu, Xianao Lu
# ID: 19990130-2035   20021201-3338

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random


# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED = '#FFC4CC';
LIGHT_GREEN = '#95FD99';
BLACK = '#000000';
WHITE = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';
LIGHT_YELLOW = '#FFFFE0'; # keys
Poison = 1 / 30;
poison_state = [(-1, -1, -1, -1)];


class Maze:
    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 10000
    IMPOSSIBLE_REWARD = -1000
    CATCH_REWARD = -1000
    KEY_REWARD = 10000

    LIFE_TIME = 50

    def __init__(self, maze, Stay=False, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze = maze;
        self.actions = self.__actions();
        self.states, self.map = self.__states();
        self.n_actions = len(self.actions);
        self.n_states = len(self.states);
        self.rewards = self.__rewards(weights=weights);
        self.STAY = Stay;
        self.s_now = 0;
        self.s_nxt = 0;
        self.key_now = 0;
        self.key_nxt = 0;

    def reset(self):
        self.s_now = self.map[(0, 0, 6, 5)];
        self.key_now = 0;
        self.s_nxt = self.map[(0, 0, 6, 5)];
        self.key_nxt = 0;
    def __actions(self):
        actions = dict();
        actions[self.STAY] = (0, 0);
        actions[self.MOVE_LEFT] = (0, -1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP] = (-1, 0);
        actions[self.MOVE_DOWN] = (1, 0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):

                if self.maze[i, j] != 1:

                    for k in range(self.maze.shape[0]):
                        for g in range(self.maze.shape[1]):
                            states[s] = (i, j, k, g);
                            map[(i, j, k, g)] = s;
                            s += 1;
        return states, map

    def get_m_action(self, state):
        # get the action of monitaur
        row = self.states[state][2];
        col = self.states[state][3];
        row_p = self.states[state][0];
        col_p = self.states[state][1];

        m_actions_list = [];

        dist_row = row_p - row;
        dist_col = col_p - col;

        ###### move towards the player
        if abs(dist_row) >= abs(dist_col):
            if dist_row > 0:
                m_actions_list.append(4);  #move down
            elif dist_row < 0:
                m_actions_list.append(3);  #move up
        else:
            if dist_col > 0:
                m_actions_list.append(2)   #move right
            elif dist_col < 0:
                m_actions_list.append(1)   #move left


        Is_edge = (row == 0) or (row == self.maze.shape[0] - 1) or \
                  (col == 0) or (col == self.maze.shape[1] - 1);
        if not Is_edge:
            if not self.STAY:
                for action in list(self.actions.keys())[1:]:
                    if not dist_row == 0 and dist_col == 0:
                        if action != m_actions_list[0]:
                            m_actions_list.append(action);
                    else:
                        m_actions_list.append(action);
            else:
                m_actions_list = list(self.actions.keys())[0:];
        else:
            if not self.STAY:
                for action in list(self.actions.keys())[1:]:
                    new_row = row + self.actions[action][0];
                    new_col = col + self.actions[action][1];

                    Is_out = (new_row == -1) or (new_row == self.maze.shape[0]) or \
                             (new_col == -1) or (new_col == self.maze.shape[1]);

                    if not Is_out:
                        if not dist_row == 0 and dist_col == 0:
                            if action != m_actions_list[0]:
                                m_actions_list.append(action);
                        else:
                            m_actions_list.append(action);
            else:
                for action in list(self.actions.keys())[0:]:
                    new_row = row + self.actions[action][0];
                    new_col = col + self.actions[action][1];

                    Is_out = (new_row == -1) or (new_row == self.maze.shape[0]) or \
                             (new_col == -1) or (new_col == self.maze.shape[1]);

                    if not Is_out:
                        m_actions_list.append(action);
        return m_actions_list;

    def __move(self, state, action, m_action=None, poison_mode=None):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # if player is dead because of poison
        if poison_mode is not None:
            seed = np.random.rand();
            if seed < Poison:
                return self.n_states;  # return the poison state, stop moving.

        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];

        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                             (col == -1) or (col == self.maze.shape[1]) or \
                             (self.maze[row, col] == 1);
        # Based on the impossibility check return the next state.
        if hitting_maze_walls:
            row = self.states[state][0];
            col = self.states[state][1];

        if m_action is not None:
            # monitaur
            row_m = self.states[state][2] + self.actions[m_action][0];
            col_m = self.states[state][3] + self.actions[m_action][1];
        else:
            m_actions_list = self.get_m_action(state);
            if np.random.random() < 0.35:   #with 0.35 probability moving towards the player
                m_action = m_actions_list[0];
                row_m = self.states[state][2] + self.actions[m_action][0];
                col_m = self.states[state][3] + self.actions[m_action][1];
            else:
                m_action = np.random.choice(m_actions_list[1:]);
                row_m = self.states[state][2] + self.actions[m_action][0];
                col_m = self.states[state][3] + self.actions[m_action][1];

        return self.map[(row, col, row_m, col_m)];


    def __rewards(self, weights=None):

        rewards = np.zeros((self.n_states, 2,self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    for key in range(2):
                        m_actions_list = self.get_m_action(s);
                        for m_action in m_actions_list:

                            # real next state, it is stochastic, because the possibility of
                            # being caught
                            next_s = self.__move(s, a, m_action);

                            # get the position of current state and next state
                            row1, col1, row1_m, col1_m = self.states[s];
                            row2, col2, row2_m, col2_m = self.states[next_s];

                            # Reward for hitting a wall
                            if [row1, col1] == [row2, col2] and a != self.STAY:
                                rewards[s, key, a] = self.IMPOSSIBLE_REWARD;

                            # Reward for getting keys
                            elif self.maze[row2, col2] == 3 and key == 0:
                                rewards[s, key, a] = self.KEY_REWARD;

                            # Be caught
                            elif [row2, col2] == [row2_m, col2_m] and self.maze[row2, col2] != 2:
                                rewards[s, key, a] = self.CATCH_REWARD / len(m_actions_list);

                            # Reward for reaching the exit
                            elif [row1, col1] == [row2, col2] and self.maze[row2, col2] == 2 and key == 1:
                                rewards[s, key, a] = self.GOAL_REWARD;

                            # Reward for taking a step to an empty cell that is not the exit
                            else:
                                rewards[s, key, a] = self.STEP_REWARD;
        return rewards;

    def check_legitimacy(self, s):
        if (self.states[s][0:2] != self.states[s][2:4] and
                self.maze[self.states[s][0], self.states[s][1]] != 1):
            flag = True;
        else:
            flag = False;
        return flag;

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);
        legitimaty_flag = True;
        success_flag = False;
        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon - 1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s, t]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                flag = self.check_legitimacy(next_s);
                if (flag == False):
                    legitimaty_flag = False;
                t += 1;
                s = next_s;

            if (legitimaty_flag == True and self.maze[path[-1][0]][path[-1][1]] == 2):
                success_flag = True;
            else:
                success_flag = False;
            # print(path);

        if method == 'ValIter':
            # Initialize current state, next state and time
            self.states[self.n_states] = poison_state[0];  # add dead by poison state
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s, policy[s], poison_mode=True);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);

            # Loop while state is not the goal state
            while self.states[s][:2] != (6, 5) and next_s != self.n_states:  # exit or not poison to dead
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s], poison_mode=True);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                flag = self.check_legitimacy(next_s, poison_mode=True);
                if (flag == False):
                    legitimaty_flag = False;
                # Update time and state for next iteration
                t += 1;
            if (legitimaty_flag == True and self.maze[path[-1][0]][path[-1][1]] == 2):
                success_flag = True;
            else:
                success_flag = False;
        return path, success_flag

    def step(self, action):
        self.s_nxt = self.__move(self.s_now, action);
        reward = self.rewards[self.s_now, self.key_now, action];
        row, col, _, _ = self.states[self.s_now];
        row_nxt, col_nxt, _, _ = self.states[self.s_nxt];
        if(self.maze[row_nxt][col_nxt] == 3):
            self.key_nxt = 1;

        legal_flag = self.check_legitimacy(self.s_now);
        if np.random.random() < 1/self.LIFE_TIME:
            poison_flag = True;
        else:
            poison_flag = False;
        if self.maze[row][col] == 2:
            exit_flag = True;
        else:
            exit_flag = False;

        if not legal_flag or poison_flag or exit_flag:
            end_flag = True;
        else:
            end_flag = False;

        # if not legal_flag or exit_flag:
        #     end_flag = True;
        # else:
        #     end_flag = False;

        self.s_now = self.s_nxt;
        self.key_now = self.key_nxt;

        return reward, self.s_nxt, self.key_nxt, end_flag;

    def show(self):
        # print('The states are :')
        # print(self.states)
        # print('The actions are:')
        # print(self.actions)
        # print('The mapping of the states:')
        # print(self.map)
        # print('The rewards:')
        print(self.rewards)
        # print(self.transition_probabilities)

class SARSA:

    def __init__(self, epsilon, alpha, lambda_para, n_states, n_actions):
        self.epsilon = epsilon;
        self.alpha = alpha;
        self.lambda_p = lambda_para;
        self.n_states = n_states;
        self.n_actions = n_actions;
        self.Q_val = np.zeros((n_states, 2, n_actions));
        self.n_vis = np.zeros((n_states, 2, n_actions));
        self.delta = 0.8

    def choose_action(self, state, key_s,action_space):
        rand_num = np.random.random();
        if rand_num < self.epsilon:
            action = np.random.choice(action_space);
        else:
            action = np.argmax(self.Q_val[state, key_s]);
        return action;

    def Q_val_update(self, s_now, key_now, action, reward, s_nxt, key_nxt, action_nxt, epslilon_num = None):
        if epslilon_num is None:
            self.n_vis[s_now, key_now, action] += 1;
            alpha = 1 / pow(self.n_vis[s_now, key_now, action], self.alpha);
            err = reward + self.lambda_p * self.Q_val[s_nxt, key_nxt, action_nxt] - self.Q_val[s_now, key_now, action];
            self.Q_val[s_now, key_now, action] = self.Q_val[s_now, key_now, action] + alpha * err;
        else:
            self.epsilon = 1 / pow(epslilon_num, self.delta);
            self.n_vis[s_now, key_now, action] += 1;
            alpha = 1 / pow(self.n_vis[s_now, key_now, action], self.alpha);
            err = reward + self.lambda_p * self.Q_val[s_nxt, key_nxt, action_nxt] - self.Q_val[s_now, key_now, action];
            self.Q_val[s_now, key_now, action] = self.Q_val[s_now, key_now, action] + alpha * err;

    def check_convergence(self):
        BV = np.zeros(self.n_states * 2);
        BV_mat = np.zeros((self.n_states, 2));
        BV_mat = np.max(self.Q_val, 2);
        BV = np.reshape(BV_mat, self.n_states * 2);
        return BV;

def SARSA_algorithm(env, episodes, epsilon, alpha, lambda_para, epslilon_mode = None):

    n_states = env.n_states;
    n_actions = env.n_actions;
    action_space = [0, 1, 2, 3, 4];
    sarsa = SARSA(epsilon, alpha, lambda_para, n_states, n_actions);

    if epslilon_mode is None:
        re = [];
        convergence = [];
        Q_init = [];
        Q_init.append(np.max(sarsa.Q_val[53, 0]));
        for i in range(episodes):
            env.reset();
            s_now = env.s_now;
            key_now = env.key_now;
            action = sarsa.choose_action(s_now, key_now, action_space);
            end = False;
            episode_return = 0;
            V = sarsa.check_convergence();
            while not end:
                reward, s_nxt, key_nxt, end = env.step(action);
                episode_return += reward;
                action_nxt = sarsa.choose_action(s_nxt, key_nxt, action_space);
                sarsa.Q_val_update(s_now, key_now, action, reward, s_nxt, key_nxt, action_nxt);
                s_now = s_nxt;
                key_now = key_nxt;
                action = action_nxt;
            BV = sarsa.check_convergence();
            convergence.append(np.linalg.norm(V - BV));
            re.append(episode_return);
            Q_init.append(np.max(sarsa.Q_val[53, 0]));

    else:
        re = [];
        convergence = [];
        Q_init = [];
        Q_init.append(np.max(sarsa.Q_val[53, 0]));
        for i in range(episodes):
            env.reset();
            s_now = env.s_now;
            key_now = env.key_now;
            action = sarsa.choose_action(s_now, key_now, action_space);
            end = False;
            episode_return = 0;
            V = sarsa.check_convergence();
            while not end:
                reward, s_nxt, key_nxt, end = env.step(action);
                episode_return += reward;
                action_nxt = sarsa.choose_action(s_nxt, key_nxt, action_space);
                sarsa.Q_val_update(s_now, key_now, action, reward, s_nxt, key_nxt, action_nxt, i+1);
                s_now = s_nxt;
                key_now = key_nxt;
                action = action_nxt;
            BV = sarsa.check_convergence();
            convergence.append(np.linalg.norm(V - BV));
            re.append(episode_return);
            Q_init.append(np.max(sarsa.Q_val[53, 0]));


    return sarsa, re, convergence, Q_init;

def sarsa_simulation(env, sarsa):
    env.reset();
    s_now = env.s_now;
    end = False;
    path = []
    while not end:
        path.append(env.states[s_now]);
        action = np.argmax(sarsa.Q_val[s_now, env.key_now]);
        _, s_nxt, key_nxt, end= env.step(action);
        s_now = s_nxt;
        # print(env.states[s_nxt]);
    return path;






def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)


def animate_solution(maze, path):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows, cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows));

    # Remove the axis ticks and add title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows);
        cell.set_width(1.0 / cols);

    # Update the color at each frame
    for i in range(len(path)):
        # if player is dead because of poison
        if path[i] == (-1, -1, -1, -1):
            grid.get_celld()[(path[i - 1][:2])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path[i - 1][2:])].set_facecolor(col_map[maze[path[i - 1][2:]]])
            grid.get_celld()[(path[i - 1][:2])].get_text().set_text('Player is dead')
            grid.get_celld()[(path[i - 1][2:])].get_text().set_text('Monitaur')
            break

        # initial the name
        grid.get_celld()[(path[i][:2])].get_text().set_text('Player')
        grid.get_celld()[(path[i][2:])].get_text().set_text('Monitaur')
        if i > 0:
            if path[i][:2] == path[i - 1][:2]:
                grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i - 1][2:])].set_facecolor(col_map[maze[path[i - 1][2:]]])
                grid.get_celld()[(path[i])[:2]].get_text().set_text('Game over')
                grid.get_celld()[(path[i - 1][2:])].get_text().set_text('')
            else:
                grid.get_celld()[(path[i - 1][2:])].set_facecolor(col_map[maze[path[i - 1][2:]]])
                grid.get_celld()[(path[i - 1][:2])].set_facecolor(col_map[maze[path[i - 1][:2]]])
                # clear the text
                grid.get_celld()[(path[i - 1][:2])].get_text().set_text('')
                grid.get_celld()[(path[i - 1][2:])].get_text().set_text('')

        grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][2:])].set_facecolor(LIGHT_PURPLE)
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(0.3)
