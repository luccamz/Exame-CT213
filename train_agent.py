import os

# to silence tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.wrappers.time_limit import TimeLimit
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from utils import reward_engineering, BOARD_SZ, TIME_LIMIT, MAP_NAME, FIXED_SEED
import tensorflow as tf

# rhis is important
tf.compat.v1.disable_eager_execution()

NUM_EPISODES = 500  # Number of episodes used for training

# generates map from the seed
map_desc = generate_random_map(size=BOARD_SZ, seed = FIXED_SEED)
env = gym.make('FrozenLake-v1', desc=map_desc, map_name=MAP_NAME, is_slippery=True)
env = TimeLimit(env, TIME_LIMIT) # generates truncated state upon reaching time limit
action_size = env.action_space.n
observation_sz = env.observation_space.n

# Creating the DQN agent
agent = DQNAgent(observation_sz, action_size, buffer_size= int(20*observation_sz))

batch_size = observation_sz  # batch size used for the experience replay
return_history = []

for episode in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state, _ = env.reset()
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    # previous time step action
    prev_action = -1 
    for time in range(1, TIME_LIMIT+1):
        # Select action
        action = agent.act(state)
        # Take action, observe reward and new state
        next_state, completed, terminated, truncated, _ = env.step(action)
        # Modifying reward
        reward = reward_engineering(state, prev_action, action, completed, next_state, terminated, truncated)
        prev_action = action
        # Appending this experience to the experience replay buffer
        agent.append_experience(state, action, reward, next_state)
        state = next_state
        # Accumulate reward
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if terminated or truncated:
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                  .format(episode, NUM_EPISODES, time, cumulative_reward, agent.epsilon))
            break
        # We only update the policy if we already have enough experience in memory
        if len(agent.replay_buffer) > 2 * batch_size:
            agent.replay(batch_size)
    return_history.append(cumulative_reward)
    agent.update_epsilon()
    if episode % 20:
        #Saving the model to disk
        agent.save("frozen_lake.pkl")

agent.save("frozen_lake.pkl")
#print(agent.q)
agent.display_greedy_policy()

plt.plot(return_history, 'b')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('dqn_training.eps', format='eps')