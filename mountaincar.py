import gym 
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import random
import matplotlib.pyplot as plt

class QNetwork(tf.keras.Model):
    """ Deep Q-Network to represent Q-Function.
    Q(s) = {Q(s, a[0]), Q(s, a[1])}
    """
    def __init__(self, env: gym.Env):
        super(QNetwork, self).__init__()
        n_states = env.observation_space.shape
        self.l_in = layers.InputLayer(input_shape=(n_states))

        n_actions = env.action_space.n
        self.l_h = [layers.Dense(20, activation='relu') for i in range(3)]

        self.l_out = layers.Dense(n_actions, activation='linear')

        self.optimizer = tf.optimizers.SGD(0.01)

    @tf.function
    def call(self, inputs):
        x = self.l_in(inputs)
        for layer in self.l_h:
            x = layer(x)
        return self.l_out(x)

    @tf.function
    def train(self, y_target, s_batch, a_batch):
        with tf.GradientTape() as tape:
            q_policy = tf.reduce_sum(self(s_batch) * tf.one_hot(a_batch, env.action_space.n), 1)
            loss = tf.math.reduce_mean(tf.square(y_target - q_policy))

        # Update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss


class Memory:
    """ Memory buffer for experience replay.
    """
    def __init__(self, size):
        # Buffer size
        self.size = size
        # Experience entries (tuples)
        self.memory = list()
        # Keep track of number of elements added
        self.counter = 0

    def add_experience(self, s: np.array, a: float, s_n: np.array, r_n: float, done: bool):
        """ Add an experience data point

        Args:
            s ([np.array]): Current state
            a ([float]): Action taken
            s_n ([np.array]): Next state
            r_n ([float]): Reward obtained
            done (boolean): Action led to finished experiment
        """
        if self.counter > self.size:
            self.memory[self.counter % self.size] = (s, a, s_n, r_n, done)
        else:
            self.memory.append((s, a, s_n, r_n, done))
        self.counter += 1

    def sample(self, num_elements: int) -> tuple:
        """ Sample num_elements experience points uniformly by random

        Args:
            num_elements (int): Number of datapoints to sample

        Returns:
            list: [description]
        """
        exp_list = random.sample(self.memory, num_elements)
        exp_list = list(map(list, zip(*exp_list)))
        return np.array(exp_list[0]), np.array(exp_list[1]), np.array(exp_list[2]), np.array(exp_list[3]), np.array(exp_list[4]) 
        

    def can_sample(self, num_elements: int) -> bool:
        return (len(self.memory) > num_elements)

class DQNActor:
    """ DQN Actor: Uses deep neural network for Q-function
    approximation. Generalized value function iteration for
    Q-function optimization.
    """

    def __init__(self, env: gym.Env):
        self.env: gym.Env = env
        self.observations = []

        # Exploration settings
        self.eps_high = 1.0
        self.eps_low = 0.0
        self.eps_decay = 0.001

        # Sampling settings
        self.n_episodes = 1000
        self.batch_size = 128
        self.memory = Memory(100000)

        # Discount factor
        self.discount_factor = 0.99

        # Q-Function settings
        self.copy_interval = 25
        self.q_policy = QNetwork(env)
        self.q_target = QNetwork(env)

        # Visualization every x episodes
        self.vis_interval = 100

    def get_epsilon(self, k: int):
        """ Get current exploration coefficient.

        Args:
            k ([int]): Number of current episode

        Returns:
            float: Epsilon for e_greedy action selection.
        """
        coef = 1.0 / max(k - 50, 1.0)
        return self.eps_low + (self.eps_high - self.eps_low) * coef

    def e_greedy(self, state: np.array, epsilon: float):
        """ Epsilon-greedy control policy

        Args:
            state ([type]): [description]
            epsilon ([type]): [description]

        Returns:
            int: Action
        """
        if np.random.rand() > epsilon:
            return np.argmax(self.q_policy(np.atleast_2d(state)))
        else:
            return self.env.action_space.sample()

    def update_target(self):
        """ Update target Q-network with current policy Q-network.
        """
        var_target = self.q_target.trainable_variables
        var_policy = self.q_policy.trainable_variables
        for (vt, vp) in zip(var_target, var_policy):
            vt.assign(vp.numpy())
    
    def init_reward_plot(self):
        """ Initialize plot of reward function
        """
        plt.ion()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        self.hl, = self.ax.plot([], [])
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Total reward per episode")

    def update_plot(self, k, r_tot):
        self.hl.set_xdata(np.append(self.hl.get_xdata(), k))
        self.hl.set_ydata(np.append(self.hl.get_ydata(), r_tot))
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def train(self):
        """ Main training loop"""
        # Make sure target and policy network are equal
        self.update_target()
        self.init_reward_plot()
        
        for k in range(self.n_episodes):
            # Reset environment to initial state
            s_curr = self.env.reset()

            # Compute exploration coefficient for current episode
            epsilon = self.get_epsilon(k)

            # Print user infos
            if (k % self.vis_interval) == 0:
                self.env.render()
            print("Epoch {}/{}, Epsilon: {}".format(k, self.n_episodes, epsilon))

            is_done = False
            r_tot = 0
            while not is_done:
                # Take epsilon-greedy action, observe state transition and reward
                a_curr = self.e_greedy(s_curr, epsilon)
                s_next, r_next, is_done, _ = self.env.step(a_curr)
                
                # Add experience to memory of the agent 
                self.memory.add_experience(s_curr, a_curr, s_next, r_next, is_done)
                if self.memory.can_sample(self.batch_size):
                    # Sample experience from past (experience replay)
                    s_batch, a_batch, s_n_batch, r_n_batch, done_batch = self.memory.sample(self.batch_size)

                    # Compute estimated q-function (bootstrapping)
                    q_target = self.q_target(s_n_batch)
                    q_target_max = tf.math.reduce_max(q_target, axis=1)
                    y_target = tf.where(done_batch, r_n_batch, r_n_batch + self.discount_factor * q_target_max)

                    # Train the model
                    self.q_policy.train(y_target, s_batch, a_batch)


                # Increment current state
                s_curr = s_next

                # Bookkeeping and visualization
                if (k % self.vis_interval) == 0:
                    self.env.render()
                r_tot += r_next

            # At fixed intervals, update target Q-network
            if (k % self.copy_interval) == 0:
                self.update_target()

            self.update_plot(k, r_tot)
            
if __name__=='__main__':
    env = gym.make('CartPole-v0')
    #env = gym.make('MountainCar-v0')
    actor = DQNActor(env)
    actor.train()
        
