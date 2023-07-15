import os
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from utils import reward_engineering
import tensorflow as tf


NUM_EPISODES = 300  # Number of episodes used for training

tf.compat.v1.disable_eager_execution()

map_desc = generate_random_map(size=4, seed = 0)
env = gym.make('FrozenLake-v1', desc=map_desc, map_name="4x4", is_slippery=True)
action_size = env.action_space.n

# Creating the DQN agent
agent = DQNAgent(action_size)

# Checking if weights from previous learning session exists
if os.path.exists('frozen_lake.h5'):
    print('Loading weights from previous learning session.')
    agent.load("frozen_lake.h5")
else:
    print('No weights found from previous learning session.')
done = False
batch_size = 4  # batch size used for the experience replay
return_history = []

for episodes in range(1, NUM_EPISODES + 1):
    print(episodes)
    # Reset the environment
    state, _ = env.reset()
    state = np.array([state])
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    for time in range(1, 100):
        # Select action
        action = agent.act(state)
        # Take action, observe reward and new state
        next_state, reward, terminated, _, _ = env.step(action)
        next_state = np.array([next_state])
        # Making reward engineering to allow faster training
        reward = reward_engineering(state, action, reward, next_state)
        # Appending this experience to the experience replay buffer
        agent.append_experience(state, action, reward, next_state, terminated)
        state = next_state
        # Accumulate reward
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if terminated:
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                  .format(episodes, NUM_EPISODES, time, cumulative_reward, agent.epsilon))
            break
        # We only update the policy if we already have enough experience in memory
        if len(agent.replay_buffer) > 2 * batch_size:
            loss = agent.replay(batch_size)
    return_history.append(cumulative_reward)
    agent.update_epsilon()
    # Every 10 episodes, update the plot for training monitoring
    if episodes % 20 == 0:
        plt.plot(return_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.show(block=False)
        plt.pause(0.1)
        plt.savefig('dqn_training.eps', format='eps')
        # Saving the model to disk
        agent.save("frozen_lake.h5")
plt.pause(1.0)