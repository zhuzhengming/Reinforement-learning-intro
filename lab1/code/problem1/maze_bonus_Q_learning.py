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
methods = ['DynProg', 'ValIter', 'Q-learning'];

# Some colours
LIGHT_RED = '#FFC4CC';
LIGHT_GREEN = '#95FD99';
BLACK = '#000000';
WHITE = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';
LIGHT_YELLOW = '#FFFFE0';  # keys
Poison = 1 / 50;
poison_state = [(-1, -1, -1, -1, -1)];


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
    GET_KEY_REWARD = 10000
    GOAL_REWARD = 10000
    IMPOSSIBLE_REWARD = -1000
    CATCH_REWARD = -1000

    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze = maze;
        self.actions = self.__actions();
        self.states, self.map = self.__states();
        self.n_actions = len(self.actions);
        self.n_states = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards = self.__rewards(weights=weights);

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
        for key in range(0, 2):
            for i in range(self.maze.shape[0]):
                for j in range(self.maze.shape[1]):

                    if self.maze[i, j] != 1:

                        for k in range(self.maze.shape[0]):
                            for g in range(self.maze.shape[1]):
                                states[s] = (i, j, k, g, key);
                                map[(i, j, k, g, key)] = s;
                                s += 1;
        return states, map

    def get_m_action(self, state):
        # get the action
        row_p, col_p, row, col = self.states[state][0:4];

        m_actions_list = [];

        # move towards the player
        dist_row = row_p - row;
        dist_col = col_p - col;
        if abs(dist_row) >= abs(dist_col):
            if dist_row > 0:
                m_actions_list.append(4);  # move down
            elif dist_row < 0:
                m_actions_list.append(3);  # move up
        else:
            if dist_col > 0:
                m_actions_list.append(2)  # move right
            elif dist_col < 0:
                m_actions_list.append(1)  # move left

        Is_edge = (row == 0) or (row == self.maze.shape[0] - 1) or \
                  (col == 0) or (col == self.maze.shape[1] - 1);
        if not Is_edge:
            for action in list(self.actions.keys())[1:]:
                if not dist_row == 0 and dist_col == 0:
                    if action != m_actions_list[0]:
                        m_actions_list.append(action);
                else:
                    m_actions_list.append(action);  # at the same place
        else:
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
                        m_actions_list.append(action);  # at the same place
        return m_actions_list;

    def move(self, state, action, m_action=None, poison_mode=False):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # if player is dead because of poison
        if poison_mode is not False:
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

        ############################################################################################

        if m_action is not None:
            # monitaur
            row_m = self.states[state][2] + self.actions[m_action][0];
            col_m = self.states[state][3] + self.actions[m_action][1];
        else:
            m_actions_list = self.get_m_action(state);
            if np.random.rand() < 0.35:  # towards player
                m_action = m_actions_list[0];
                row_m = self.states[state][2] + self.actions[m_action][0];
                col_m = self.states[state][3] + self.actions[m_action][1];
            else:
                m_action = np.random.choice(m_actions_list[1:]);
                row_m = self.states[state][2] + self.actions[m_action][0];
                col_m = self.states[state][3] + self.actions[m_action][1];

        if [row, col] == [0,7] and self.states[state][4] == 0:
            key = 1;
        else:
            key = self.states[state][4];


        return self.map[(row, col, row_m, col_m, key)];

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probabilities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        # fixed state
        for s in range(self.n_states):
            m_actions_list = self.get_m_action(s);

            # towards player
            for a in range(self.n_actions):
                next_s = self.move(s, a, m_actions_list[0]);
                transition_probabilities[next_s, s, a] = 0.35;

            # randomly
            for m_action in m_actions_list[1:]:
                # for every action
                for a in range(self.n_actions):
                    next_s = self.move(s, a, m_action);
                    N_m_actions = len(m_actions_list);
                    transition_probabilities[next_s, s, a] = 0.65 / (N_m_actions - 1);
        return transition_probabilities;

    def __rewards(self, weights=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):

                    # real next state, it is stochastic, because the possibility of
                    # being caught
                    next_s = self.move(s, a);

                    # Can be caught or not
                    Is_caught = False;
                    m_actions_list = self.get_m_action(s);
                    for m_action in m_actions_list:
                        next_M_s = self.move(s, a, m_action);
                        row, col, row_m, col_m = self.states[next_M_s][0:4];
                        if [row, col] == [row_m, col_m]:
                            Is_caught = True;

                    # get the position of current state and next state
                    row1, col1, row1_m, col1_m, key1 = self.states[s];
                    row2, col2, row2_m, col2_m, key2 = self.states[next_s];

                    # Reward for hitting a wall
                    if [row1, col1] == [row2, col2] and a != self.STAY:
                        rewards[s, a] = self.IMPOSSIBLE_REWARD;

                    # getting key
                    elif self.maze[row2, col2] == 3 and key1 == 0:
                        rewards[s, a] = self.GET_KEY_REWARD;

                    # Be caught
                    elif Is_caught and self.maze[row2, col2] != 2:
                        rewards[s, a] = self.CATCH_REWARD / len(m_actions_list);

                    # Reward for reaching the exit
                    elif self.maze[row2, col2] == 2 and key1 == 1:
                        rewards[s, a] = self.GOAL_REWARD;

                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s, a] = self.STEP_REWARD;

        return rewards;

    def check_legitimacy(self, s, poison_mode=None):
        if poison_mode is None:
            if (self.states[s][0:2] != self.states[s][2:4] and
                    self.maze[self.states[s][0], self.states[s][1]] != 1):
                return True;
            else:
                return False;
        else:
            if (self.states[s][0:2] != self.states[s][2:4] and
                    self.maze[self.states[s][0], self.states[s][1]] != 1 and
                    self.states[s] != poison_state[0]):
                return True;
            else:
                return False;

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);
        legitimaty_flag = True;
        success_flag = False;
        path = list();
        if method == 'Q-learning':
            # Initialize current state, next state and time
            self.states[self.n_states] = poison_state[0];  # add dead by poison state
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            # next_s = self.move(s, policy[s], poison_mode=True);
            # # Add the position in the maze corresponding to the next state
            # # to the path
            # path.append(self.states[next_s]);

            # Loop while state is not the goal state
            while True:
                # Update state
                # Move to next state given the policy and the current state
                next_s = self.move(s, policy[s], poison_mode=True);
                s = next_s;
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                flag = self.check_legitimacy(next_s, poison_mode=True);
                if (flag == False):
                    legitimaty_flag = False;
                # Update time and state for next iteration
                t += 1;
                row, col, row_m, col_m, key = self.states[next_s];
                if (self.states[next_s] == poison_state[0] or
                        (self.maze[row, col] == 2 and key) or
                        [row, col] == [row_m, col_m]):
                    break

            if (legitimaty_flag == True and self.maze[path[-1][0]][path[-1][1]] == 2):
                success_flag = True;
            else:
                success_flag = False;
        return path, success_flag

    def show(self):
        # print('The states are :')
        # print(self.states)
        # print('The actions are:')
        # print(self.actions)
        # print('The mapping of the states:')
        # print(self.map)
        print('The rewards:')
        print(self.rewards)
        print(self.transition_probabilities)


def epsilon_greedy(state, Q, epsilon):
    if np.random.uniform(0,1) < epsilon:
        actions_list = [0, 1, 2, 3, 4]
        return np.random.choice(actions_list)
    else:
        return np.argmax(Q[state, :])  # choose the action which make Q biggest


def Q_learning(env, epsilon, num_episodes, gamma, start, alpha, poison_mode=False):
    # epsilon: ε-greedy
    # gamma: discount
    # lr: learning rate =  1/n(s, a)^2/3

    POISON_REWARD = -1000
    if poison_mode is not False:
        # modify MDP because fo additional state "poison"
        states = np.array(list(env.map.keys()) + poison_state);  # add the state poison
        n_states = states.shape[0];
        n_actions = env.n_actions;

        r = np.zeros((n_states, n_actions));
        r[:-1, :] = env.rewards;  # alive: r'(s,a) = r(s,a)
        r[-1, :] = POISON_REWARD;  # dead r'(s,a) = POISON_REWARD
    else:
        states = np.array(list(env.map.keys()))
        r = env.rewards;
        n_states = env.n_states;
        n_actions = env.n_actions;

    # Initial Q value:
    Q = np.zeros((n_states, n_actions));
    visit_count = np.zeros((n_states, n_actions));
    td_errors = []
    V_start = []

    for episode in range(num_episodes):
        state = env.map[start];  # initial state
        total_TD_error = 0;
        count = 0;

        while True:
            # select action according to ε-greedy
            action = epsilon_greedy(state, Q, epsilon);
            # update next state
            next_s = env.move(state, action, poison_mode=True);
            # record visit (r,a)
            visit_count[state, action] += 1
            # calculate learning rate
            lr = 1 / (visit_count[state, action] ** alpha);

            td_error = r[state, action] + gamma * np.max(Q[next_s, :]) - Q[state, action];
            total_TD_error += abs(td_error);
            count += 1;
            # update Q-value
            Q[state, action] = Q[state, action] + lr * td_error;
            state = next_s

            # terminal state:
            # poison
            # exit
            # eaten
            row, col, row_m, col_m, key = states[state];
            if state == n_states or (env.maze[row, col] == 2 and key) or [row, col] == [row_m, col_m]:
                break
        avg_td_array = total_TD_error / count;
        td_errors.append(avg_td_array);
        V_start.append(np.max(Q[env.map[start], :]))
        # print(episode)
    # print(Q[env.map[start], :])
    policy = np.argmax(Q, 1);
    return td_errors, policy, V_start;

def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: LIGHT_YELLOW, -6: LIGHT_RED, -1: LIGHT_RED}

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
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: LIGHT_YELLOW, -6: LIGHT_RED, -1: LIGHT_RED};

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
        if path[i][0:4] == (-1, -1, -1, -1):
            grid.get_celld()[(path[i - 1][:2])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path[i - 1][2:4])].set_facecolor(col_map[maze[path[i - 1][2:4]]])
            grid.get_celld()[(path[i - 1][:2])].get_text().set_text('Player is dead')
            grid.get_celld()[(path[i - 1][2:4])].get_text().set_text('Monitaur')
            break

        # initial the name
        grid.get_celld()[(path[i][:2])].get_text().set_text('Player')
        grid.get_celld()[(path[i][2:4])].get_text().set_text('Monitaur')
        if i > 0:
            if path[i][:2] == path[i - 1][:2]:
                grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i - 1][2:4])].set_facecolor(col_map[maze[path[i - 1][2:4]]])
                grid.get_celld()[(path[i - 1][2:4])].get_text().set_text('')
            else:
                grid.get_celld()[(path[i - 1][2:4])].set_facecolor(col_map[maze[path[i - 1][2:4]]])
                grid.get_celld()[(path[i - 1][:2])].set_facecolor(col_map[maze[path[i - 1][:2]]])
                # clear the text
                grid.get_celld()[(path[i - 1][:2])].get_text().set_text('')
                grid.get_celld()[(path[i - 1][2:4])].get_text().set_text('')

        grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][2:4])].set_facecolor(LIGHT_PURPLE)
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(0.3)
