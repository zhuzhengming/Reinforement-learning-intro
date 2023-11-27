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
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100
    CATCH_REWARD = -100

    def __init__(self, maze, Stay=False, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze = maze;
        self.actions = self.__actions();
        self.states, self.map = self.__states();
        self.n_actions = len(self.actions);
        self.n_states = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards = self.__rewards(weights=weights,
                                      random_rewards=random_rewards);
        self.STAY = Stay;
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

        m_actions_list = [];

        Is_edge = (row == 0) or (row == self.maze.shape[0] - 1) or \
                  (col == 0) or (col == self.maze.shape[1] - 1);
        if not Is_edge:
            if not self.STAY:
                m_actions_list = list(self.actions.keys())[1:];
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
            m_action = np.random.choice(m_actions_list);
            row_m = self.states[state][2] + self.actions[m_action][0];
            col_m = self.states[state][3] + self.actions[m_action][1];

        return self.map[(row, col, row_m, col_m)];

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
            for m_action in m_actions_list:
                # for every action
                for a in range(self.n_actions):
                    next_s = self.__move(s, a, m_action);
                    N_m_actions = len(m_actions_list);
                    transition_probabilities[next_s, s, a] = 1 / N_m_actions;
        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):

                    # real next state, it is stochastic, because the possibility of
                    # being caught
                    next_s = self.__move(s, a);

                    # Can be caught or not
                    Is_caught = False;
                    m_actions_list = self.get_m_action(s);
                    for m_action in m_actions_list:
                        next_M_s = self.__move(s, a, m_action);
                        row, col, row_m, col_m = self.states[next_M_s];
                        if [row, col] == [row_m, col_m]:
                            Is_caught = True;

                    # get the position of current state and next state
                    row1, col1, row1_m, col1_m = self.states[s];
                    row2, col2, row2_m, col2_m = self.states[next_s];

                    # Reward for hitting a wall
                    if [row1, col1] == [row2, col2] and a != self.STAY:
                        rewards[s, a] = self.IMPOSSIBLE_REWARD;

                    # Be caught
                    elif Is_caught and self.maze[row2, col2] != 2:
                        rewards[s, a] = self.CATCH_REWARD / len(m_actions_list);

                    # Reward for reaching the exit
                    elif [row1, col1] == [row2, col2] and self.maze[row2, col2] == 2:
                        rewards[s, a] = self.GOAL_REWARD;

                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s, a] = self.STEP_REWARD;


#####################################################################################

                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s]] < 0:
                        row, col, row_m, col_m = self.states[next_s];
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s, a];
                        # With probability 0.5 the reward is
                        r2 = rewards[s, a];
                        # The average reward
                        rewards[s, a] = 0.5 * r1 + 0.5 * r2;
        # If the weights are described by a weight matrix
        else:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s, a);
                    i, j = self.states[next_s];
                    # Simply put the reward as the weights o the next state.
                    rewards[s, a] = weights[i][j];

        return rewards;

    def check_legitimacy(self, s, poison_mode=None):
        if poison_mode is None:
            if (self.states[s][0:2] != self.states[s][2:4] and
                    self.maze[self.states[s][0], self.states[s][1]] != 1):
                flag = True;
            else:
                flag = False;
        else:
            if (self.states[s][0:2] != self.states[s][2:4] and
                    self.maze[self.states[s][0], self.states[s][1]] != 1 and
                    self.states[s] != poison_state[0]):
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


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic programming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities;
    r = env.rewards;
    n_states = env.n_states;
    n_actions = env.n_actions;
    T = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T + 1));
    policy = np.zeros((n_states, T + 1));
    Q = np.zeros((n_states, n_actions));

    # Initialization
    Q = np.copy(r);
    V[:, T] = np.max(Q, 1);
    policy[:, T] = np.argmax(Q, 1);

    # The dynamic programming backwards recursion
    for t in range(T - 1, -1, -1):
        # Update the value function according to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t + 1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1);
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1);
    return V, policy;


def value_iteration(env, gamma, epsilon, poison_mode=None):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value iteration algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    POISON_REWARD = -1000

    if poison_mode != None:
        states = np.array(list(env.map.keys()) + poison_state);  # add the state poison
        n_states = states.shape[0];
        n_actions = env.n_actions;

        p = np.zeros((n_states, n_states, n_actions));
        p[:-1, :-1, :] = env.transition_probabilities * (1 - Poison);  # alive: p'(s'|s,a) = (1-1/30)*p(s'|s,a)
        p[-1, :, :] = Poison;  # dead: p'(s'|s,a) = 1/30

        r = np.zeros((n_states, n_actions));
        r[:-1, :] = env.rewards;  # alive: r'(s,a) = r(s,a)
        r[-1, :] = POISON_REWARD;  # dead r'(s,a) = POISON_REWARD

    else:
        p = env.transition_probabilities;
        r = env.rewards;
        n_states = env.n_states;
        n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states);
    Q = np.zeros((n_states, n_actions));
    BV = np.zeros(n_states);
    # Iteration counter
    n = 0;
    # Tolerance error
    tol = (1 - gamma) * epsilon / gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V);
        BV = np.max(Q, 1);
        # Show error
        print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1);
    # Return the obtained policy
    return V, policy;


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
        time.sleep(0.1)
