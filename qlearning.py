import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
SEED = 2


class Agent:
    def __init__(self, env):
        self.env: gym.Env = env
        self.Q: np.array = 0.0 * np.ones((self.env.observation_space.n, self.env.action_space.n))

    def plot_moving_average(self, rewards: np.array, win_size=40):
        n_episodes = rewards.shape[0]

        fig, ax = plt.subplots(nrows=2, ncols=1)
        k = np.arange(start=0, stop = n_episodes)

        ax[0].plot(k, rewards)
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("Reward")

        ma = np.convolve(rewards, np.ones(win_size) / win_size, mode='same')
        ax[1].plot(k, ma)
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Moving average reward (k={})".format(win_size))
        plt.show()

    def play(self):
        s = self.env.reset()
        self.env.render()
        is_done = False
        while not is_done:
            input("Press <enter> to continue")
            a = np.argmax(self.Q[s, :])
            s_next, r, is_done, _ = self.env.step(a)
            self.env.render()
            s = s_next

    def print_policy(self):
        actions = []
        for i in range(self.Q.shape[0]):
            action = np.argmax(self.Q[i, :])
            if action == 0:
                actions.append("L")
            elif action == 1:
                actions.append("D")
            elif action == 2:
                actions.append("R")
            elif action == 3:
                actions.append("U")

        print("")
        n = int(np.sqrt(self.env.observation_space.n))
        for i in range(n):
            print("{}".format(actions[(i * n) : ((i + 1) * n)]))

    def e_greedy(self, epsilon: float, s: int) -> int:
        if np.random.rand() > epsilon:
            return np.argmax(self.Q[s, :])
        else:
            return self.env.action_space.sample()

class MCAgent(Agent):
    """ Actor that learns Q-Function through MC-Sampling

    Each episode, we estimate E[G_t | S_t], by sampling from the MDP
    given an epsilon-greedy policy. This gives us estimates of Q, that minimize
    mean-squared error from the actual returns observed in the training set.
    We choose the epsilon-greedy policy w.r.t. Q. (~Policy Iteration)

    Problem here: Estimator is high in variance. Under a stochastic MDP
    this makes estimation harder!

    Args:
        Agent ([type]): [description]
    """
    def __init__(self, env):
        super().__init__(env)

    def train(self, lr: callable, discount: float, n_episodes: int, eps: callable) -> np.array:
        np.random.seed(SEED)
        rewards = np.zeros(n_episodes)
        for k in range(n_episodes):
            # Sample an instance of the MDP
            epsilon = eps(k)
            s = self.env.reset()
            sa_visited = list()
            is_done = False
            while not is_done:
                a = self.e_greedy(epsilon, s)
                sa_visited.append((s, a))
                s, r, is_done, _ = self.env.step(a)

            # Estimate E[G_t | S_t] with empirical mean
            gamma = discount
            for (s, a) in reversed(sa_visited):
                self.Q[s, a] = self.Q[s, a] + lr(k) * (gamma * r - self.Q[s, a])
                gamma *= gamma

            rewards[k] = r
        return rewards

class SarsaAgent(Agent):
    """ Actor that learns optimal Q-Function through SARSA

    Each step, we estimate E[r + gamma * Q(s', r') | s] using
    sampling and bootstrapping. This gives us estimates of Q that maximize
    log-likelihood of the underlying MDP that generated the samples. We
    chose the epsilon greedy policy w.r.t. Q.

    This estimator is low in variance, because we only randomness
    comes from r (we use the model to estimate Q(s', r')). This
    works better under stochastic MDPs. Note, that the estimator of
    Q(s, r), is biased. However, we can still guarantee convergence
    to the true action-value function. (~Policy Iteration)

    Args:
        Agent ([type]): [description]
    """
    def __init__(self, env):
        super().__init__(env)
        
    def train(self, lr: callable, discount: float, n_episodes: int, epsilon_fcn: callable) -> None:
        rewards = np.zeros(n_episodes)

        for k in range(n_episodes):
            # Decreasing greedyness
            epsilon: float = epsilon_fcn(k)

            # Reset states, initial action
            s = self.env.reset()
            a = self.e_greedy(epsilon, s)
            is_done = False
            while not is_done:
                s_next, r, is_done, _ = self.env.step(a)
                a_next = self.e_greedy(epsilon, s_next)

                self.Q[s, a] = self.Q[s,a] + lr(k) * (r + discount * self.Q[s_next, a_next] - self.Q[s, a])
                s = s_next
                a = a_next

            rewards[k] = r

        return rewards

class QLearningAgent(Agent):
    """ Actor that learns the optimal Q-Function through Q-learning

    This is somewhat similar to Value Function Iteration. Each step, 
    we estimate E[r + gamma * maxQ(s', :)] using a one sample estimate
    and bootstrapping. The learned action-value function Q directly
    approximates q_star, independent of the policy being followed.

    In Q-Learning, we only require (s, a, r, s_next) for learning. Thus,
    this is an off-policy control method.

    Args:
        Agent ([type]): [description]
    """
    def __init__(self, env):
        super().__init__(env)

    def train(self, lr: float, discount: float, n_episodes: int, eps: callable) -> np.array:
        np.random.seed(SEED)

        rewards = np.zeros(n_episodes)

        for k in range(n_episodes):
            # Epsilon decays --> RM-Conditions 
            epsilon = eps(k)
            s = self.env.reset()
            is_done = False
            while not is_done:
                # Choose action epsilon-greedily w.r.t. current Q
                action = self.e_greedy(epsilon, s)

                # Take action A, observe R, S'
                s_next, reward, is_done, _ = self.env.step(action)

                # Now we have dataset (S, A, R, S') -> Q-Update
                self.Q[s, action] = self.Q[s, action] * (1.0 - lr) + lr * (reward + discount * np.max(self.Q[s_next, :]))

                # Update state
                s = s_next

            rewards[k] = reward

        return rewards




if __name__=='__main__':
    # Policy iteration using MC-Estimation of Q
    if False:
        env = gym.make('FrozenLake-v1', is_slippery = False)

        agent = MCAgent(env)
        eps = lambda k: 0.8 / max(k - 1000, 1.0)
        lr = lambda k: 0.01
        rewards = agent.train(lr, 0.99, 2000, eps)
        agent.plot_moving_average(rewards, win_size=20)
        agent.print_policy()

    # SARSA 
    if False:
        env = gym.make('FrozenLake-v1', is_slippery = False)
        agent = SarsaAgent(env)
        eps = lambda k: 1.0 / max(k - 500, 1)
        lr = lambda k: 0.1 / max(k - 500, 1)
        rewards = agent.train(lr, 0.99, 1000, eps)
        agent.plot_moving_average(rewards, win_size=20)
        agent.print_policy()

        env = gym.make('FrozenLake-v1', is_slippery = True)
        agent = SarsaAgent(env)
        eps = lambda k: 0.6 / max(k - 1000, 1)
        lr = lambda k: 0.2 / max(k - 2000, 1)
        rewards = agent.train(lr, 0.99, 2000, eps)
        agent.plot_moving_average(rewards, win_size=20)
        agent.print_policy()

    # Q-Learning 
    if True:
        env = gym.make('FrozenLake-v1', is_slippery = True)
        agent = QLearningAgent(env)
        eps = lambda k: 1.0 / max(k - 500, 1)
        rewards = agent.train(0.2, 0.99, 2000, eps)
        agent.plot_moving_average(rewards, win_size=20)
        agent.print_policy()
