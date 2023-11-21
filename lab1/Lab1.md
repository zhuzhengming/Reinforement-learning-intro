## Lab1 

(a)
$$
State:S = \{(x,y),(x_m,y_m) \}\in R^{4N},(x,y)\notin \{walls\} \\
Action: A =\{ up, down, left, right, stay\} \\
Transition:P(s'|s,a) = 1\quad no wall \\
P(s|s,a) = 1\quad wall\\
Rewards:r(s,a) = -\infin\quad wall \\
r(s,a) = -\infin \quad catch \\
r(s,a) = -1 \quad move \\
r(s,a) = 0 \quad exit \\
$$

code：牛头人在靠墙的时候的move