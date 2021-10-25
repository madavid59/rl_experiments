import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
SEED = 2


class Agent:
    def __init__(self, env):
        self.env: gym.Env = env
        self.Q: np.array = np.zeros((self.env.observation_space.n, self.env.action_space.n))

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

class QLearningAgent(Agent):
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

class SarsaAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        
    def train(self, lr: float, discount: float, n_episodes: int, epsilon_fcn: callable) -> None:
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

                self.Q[s, a] = self.Q[s,a] + lr * (r + discount * self.Q[s_next, a_next] - self.Q[s, a])
                s = s_next
                a = a_next

            rewards[k] = r

        return rewards


if __name__=='__main__':

    """
    Q-Learning:
    Similar to value function evaluation:
    - MC-Sampling instead of expected Q-Value
    - Bootstrapping instead of MC-Sampling
    """

    # Environment: Frozen Lake
    # Case 1: Deterministic Q-Learning
    # - learning rate 1.0 yields optimal result
    if False:
        env = gym.make('FrozenLake-v1', is_slippery = False)
        agent = QLearningAgent(env)
        rewards = agent.train(lr = 1.0, discount = 0.99, n_episodes=2000)
        agent.plot_moving_average(rewards, win_size=20)
        agent.print_policy()

    # Environment: Frozen Lake
    # Case 2: Stochastic Q-Learning
    # - learning rate 1.0 does not converge due to stochasticity
    
    if False:
        env = gym.make('FrozenLake-v1', is_slippery = True)
        agent = QLearningAgent(env)
        eps = lambda k: 1.0 / max(k - 500, 1)
        rewards = agent.train(0.2, 0.99, 2000, eps)
        agent.plot_moving_average(rewards, win_size=20)
        agent.print_policy()

    if False:
        env = gym.make('FrozenLake8x8-v1', is_slippery = True)
        agent = QLearningAgent(env)
        eps = lambda k: 1.0 / max(k - 1500, 1)
        rewards = agent.train(0.30, 0.99, 5000, eps)
        agent.plot_moving_average(rewards, win_size=20)
        agent.print_policy()

    """
    TD(0)- Learning:
    Similar to policy iteration:
    - MC-Sampling instead of expected Q-Value
    - Bootstrapping instead of MC-Sampling
    """
    if True:
        env = gym.make('FrozenLake-v1', is_slippery = True)
        agent = SarsaAgent(env)
        eps = lambda k: 1.0 / max(k - 500, 1)
        rewards = agent.train(0.4, 0.99, 5000, eps)
        agent.plot_moving_average(rewards, win_size=20)
        agent.print_policy()

    