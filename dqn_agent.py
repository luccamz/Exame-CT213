import random
import numpy as np
from collections import deque
from utils import action_dir, BOARD_SZ, TIME_LIMIT

class DQNAgent:
    """
    Represents a Deep Q-Networks (DQN) agent.
    """
    def __init__(self, observation_sz, action_size, state_size = 1, alpha = 0.8, gamma=0.98, epsilon=0.8, epsilon_min=0.01, epsilon_decay=0.99, buffer_size=TIME_LIMIT):
        """
        Creates a Deep Q-Networks (DQN) agent.

        :param state_size: number of dimensions of the feature vector of the state.
        :type state_size: int.
        :param action_size: number of actions.
        :type action_size: int.
        :param gamma: discount factor.
        :type gamma: float.
        :param epsilon: epsilon used in epsilon-greedy policy.
        :type epsilon: float.
        :param epsilon_min: minimum epsilon used in epsilon-greedy policy.
        :type epsilon_min: float.
        :param epsilon_decay: decay of epsilon per episode.
        :type epsilon_decay: float.
        :param learning_rate: learning rate of the action-value neural network.
        :type learning_rate: float.
        :param buffer_size: size of the experience replay buffer.
        :type buffer_size: int.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = np.zeros((observation_sz, action_size), dtype=np.float64)

    def act(self, state):
        """
        Chooses an action using an epsilon-greedy policy.

        :param state: current state.
        :type state: NumPy array with dimension (1, 2).
        :return: chosen action.
        :rtype: int.
        """
        # redundant early return for performance
        if self.epsilon == 0.0:
            return np.argmax(self.q[state,:])
        r = random.uniform(0,1)
        if r < self.epsilon:
            return int(np.random.uniform(high=self.action_size))
        else:
            return np.argmax(self.q[state,:])

    def update(self):
        """
        Learns from memorized experience.

        :param batch_size: size of the minibatch taken from the replay buffer.
        :type batch_size: int.
        :return: loss computed during the neural network training.
        :rtype: float.
        """
        state, action, reward, next_state = self.replay_buffer[-1]
        self.q[state, action] += self.alpha*(reward + self.gamma * np.max(self.q[next_state, :]) - self.q[state, action])

    def append_experience(self, state, action, reward, next_state):
        """
        Appends a new experience to the replay buffer (and forget an old one if the buffer is full).

        :param state: state.
        :type state: NumPy array with dimension (1, 2).
        :param action: action.
        :type action: int.
        :param reward: reward.
        :type reward: float.
        :param next_state: next state.
        :type next_state: NumPy array with dimension (1, 2).
        """
        self.replay_buffer.append((state, action, reward, next_state))

    def replay(self, batch_size):
        """
        Learns from memorized experience.

        :param batch_size: size of the minibatch taken from the replay buffer.
        :type batch_size: int.
        :return: loss computed during the neural network training.
        :rtype: float.
        """
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state in minibatch:
            self.q[state, action] += self.alpha*(reward + self.gamma * np.max(self.q[next_state, :]) - self.q[state, action])

    def load(self, name):
        """
        Loads the neural network's weights from disk.

        :param name: model's name.
        :type name: str.
        """
        with open(name, "rb") as f:
            self.q = np.load(f, allow_pickle=True)

    def save(self, name):
        with open(name, "wb") as f:
            self.q.dump(f)

    def update_epsilon(self):
        """
        Updates the epsilon used for epsilon-greedy action selection.
        """
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
    
    def display_greedy_policy(self):
        disp = []
        for row in self.q:
            disp.append(action_dir[row.argmax()])
        disp = np.reshape(disp, (BOARD_SZ, BOARD_SZ))
        print("Greedy policy:")
        print(disp)