## Reinforcement learning and Implementation

- [x] MDP：建模(State space + Action space + Transition probability + Reward)

  求解：Dynamic programming，Value iteration，Policy iteration

- [x] Q-learning：关键更新公式如下，可以是off-policy也可以是on-policy
  $$Q_{new}(s,a) = Q(s,a)+\alpha[r + \gamma \max_{a'}Q(s',a')-Q(s,a)]$$

- [x] Sarsa:关键更新公式如下，on-policy:
  $$
  Q_{new}(s,a) = Q(s,a)+\alpha [r + \gamma Q(s',a')-Q(s,a)]
  $$
  注：Q-learning和Sarsa的区别在于对更新过程选择的目标Q值，前者选择下一状态的最大Q值，后者根据当前策略选择下一状态的Q值。

- [x] Sarsa(λ): 引入eligibility traces来调整短期记忆机制

- [x] Linear function approximators:将价值函数表示用一组线性函数基表示，如傅里叶基
  $$
  V_\omega(s) = \omega^T \phi(s)
  $$
  通过随机梯度下降等方法更新参数ω

- [x] DQN：引入神经网络来非线性近似Q-value，只适用于离散的动作空间。两个网络，在线网络和目标网络，使用半梯度下降方法滞后更新，损失函数定义为：
  $$
  L(\theta) = E_{s,a,r,s'}[(r+\gamma \max_{a'} Q(s',a';\theta_{copy})-Q(s,a;\theta))^2]
  $$
  半梯度更新数学表达式：
  $$
  \theta_{t+1} = \theta_t + \alpha[r+\gamma \max_{a'}Q(s',a';\theta_{copy}) - Q(s,a;\theta)]\nabla_{\theta}Q(s,a;\theta)
  $$

- [ ] DDPG: 基于Critic和Actor框架的算法，适用于连续的动作空间，训练框架基于DQN（target network，经验回放）

  Critic：评估当前动作的价值并更新，这部分本质上和DQN网络一致，这里选择的下个状态的Q value是基于actor网络基于下个状态输出的最佳动作，因为Critic网络目的是学习能够准备估计当前策略下的预期回报，从而指导actor进行策略更新。
  $$
  Q_{new}(s,a) = Q(s,a) + \alpha [r + \gamma Q'(s',\mu'(s')) - Q(s,a)] \\
  where \quad \mu'(s')\text{ is the optimal action from actor in the state } s'
  $$
  Actor：针对每个状态给出最佳动作，动作空间是连续的且是deterministic的，目的是最大化当前状态下的Q-value。
  $$
  J(\theta) = E_{s\to \rho_{\theta}}[r(s,\pi_{\theta}(s))] \\
  where \quad \pi_{\theta}(s) \text{ is trajectory under policy and state } \pi_{\theta}(s) \\
  \rho_\theta \text{ is discounted stationary distribution from the policy}\\
  \nabla_\theta J(\theta) = E_{s_t \to \rho_\theta}[\nabla_aQ(s,a)|_{a=\pi_\theta(s_t),s=s_t}\nabla_\theta \pi_\theta(s)|_{s=s_t}]\\
  where \quad \text{Q is evaluated by critic network} \\
  \theta = \theta + \alpha \nabla_\theta J(\theta)
  $$
  Noise model for exploration:
  $$
  a_t = \pi_\theta(s_t) + n_t \\
  \text{Ornstein-Uhlenbeck process }n_t = \mu n_{t-1} + \omega_{t-1}
  $$
  

- [ ] PPO

