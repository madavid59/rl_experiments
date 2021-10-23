import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(2)

class QTableAgent:
    def __init__(self, env, lr = 0.8, discount = 0.95, n_episodes = 2000):
        self.lr = lr
        self.discount = discount
        self.n_episodes = n_episodes
        self.env = env
        # Initialize Q-Table with zeros
        self.Q = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        self.rewards = np.zeros(self.n_episodes)
    

    def train(self):

        
        for k in range(self.n_episodes):
            # Epsilon decays --> RM-Conditions
            epsilon = self.n_episodes / 100 / float(k + 1)
            s = env.reset()
            is_done = False
            while not is_done:
                # Choose action epsilon-greedily w.r.t. current Q
                if np.random.rand() > epsilon:
                    action = np.argmax(self.Q[s, :])
                else:
                    action = np.random.randint(0, high=env.action_space.n)

                # Take action A, observe R, S'
                s_next, reward, is_done, _ = self.env.step(action)

                # Now we have dataset (S, A, R, S') -> Q-Update
                self.Q[s, action] = self.Q[s, action] * (1.0 - self.lr) +  self.lr * (reward + self.discount * np.max(self.Q[s_next, :]))

                # Update state
                s = s_next

            self.rewards[k] = reward

    def plot_moving_average(self, win_size=40):
        fig, ax = plt.subplots(nrows=2, ncols=1)
        k = np.arange(start=0, stop = self.n_episodes)
        ax[0].plot(k, self.rewards)
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("Reward")

        ma = np.convolve(self.rewards, np.ones(win_size) / win_size, mode='same')
        ax[1].plot(k, ma)
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Moving average reward (k={})".format(win_size))
        plt.show()


    def play(self):
        s = env.reset()
        env.render()
        is_done = False
        while not is_done:
            input("Press <enter> to continue")
            a = np.argmax(self.Q[s, :])
            s_next, r, is_done, _ = env.step(a)
            env.render()
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
        print("{}{}{}{}".format(*actions[0:4]))
        print("{}{}{}{}".format(*actions[4:8]))
        print("{}{}{}{}".format(*actions[8:12]))
        print("{}{}{}{}".format(*actions[12:16]))


if __name__=='__main__':
    # Environment: Frozen Lake
    env = gym.make('FrozenLake-v1', is_slippery = True)

    # Train Q-Table
    # A note on hyper parameter selection
    # learning rate is chosen low, because the problem is stochastic. Thus, we don't want noise
    # to mess up our value function too badly.
    agent = QTableAgent(env, lr = 0.1, discount=0.95, n_episodes=10000)
    agent.train()
    agent.plot_moving_average(win_size=20)
    agent.print_policy()
    #agent.play()
