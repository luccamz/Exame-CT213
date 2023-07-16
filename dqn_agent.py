import random
import numpy as np
from collections import deque
from tensorflow import keras
from keras import models, layers, activations
from tensorflow import optimizers, losses
from utils import BOARD_SZ

class DQNAgent:
    """
    Represents a Deep Q-Networks (DQN) agent.
    """
    def __init__(self, action_size, state_size = 1, gamma=0.95, epsilon=0.7, epsilon_min=0.01, epsilon_decay=0.98, learning_rate=0.001, buffer_size=200*BOARD_SZ):
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
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self.make_model()

    def make_model(self):
        """
        Makes the action-value neural network model using Keras.

        :return: action-value neural network.
        :rtype: Keras' model.
        """
        model = models.Sequential()
        model.add(layers.Dense(24, activation = activations.linear, input_dim = self.state_size))
        model.add(layers.ReLU())
        model.add(layers.Dense(24, activation = activations.linear))
        model.add(layers.ReLU())
        model.add(layers.Dense(self.action_size, activation = activations.linear))
        model.compile(loss=losses.mse,
                       optimizer=optimizers.legacy.Adam(learning_rate = self.learning_rate))
        return model

    def act(self, state):
        """
        Chooses an action using an epsilon-greedy policy.

        :param state: current state.
        :type state: NumPy array with dimension (1, 2).
        :return: chosen action.
        :rtype: int.
        """
        values = self.model.predict(state)[0]
        r = random.uniform(0,1)
        if r < self.epsilon:
            return int(np.random.uniform(high=self.action_size))
        else:
            return np.argmax(values)

    def append_experience(self, state, action, reward, next_state, completed):
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
        :param completed: if the end goal was reached
        :type completed: int.
        """
        self.replay_buffer.append((state, action, reward, next_state, completed))

    def replay(self, batch_size):
        """
        Learns from memorized experience.

        :param batch_size: size of the minibatch taken from the replay buffer.
        :type batch_size: int.
        :return: loss computed during the neural network training.
        :rtype: float.
        """
        minibatch = random.sample(self.replay_buffer, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, terminated in minibatch:
            target = self.model.predict(state)
            if not terminated:
                target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            else:
                target[0][action] = reward
            # Filtering out states and targets for training
            states.append(state)
            targets.append(target[0])
        history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        return loss

    def load(self, name):
        """
        Loads the neural network's weights from disk.

        :param name: model's name.
        :type name: str.
        """
        self.model.load_weights(name)

    def save(self, name):
        """
        Saves the neural network's weights to disk.

        :param name: model's name.
        :type name: str.
        """
        self.model.save_weights(name)

    def update_epsilon(self):
        """
        Updates the epsilon used for epsilon-greedy action selection.
        """
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min