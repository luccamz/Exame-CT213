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

# this line is important
tf.compat.v1.disable_eager_execution()

NUM_EPISODES = 20  # Number of episodes used for evaluation
RENDER = True # choose whether to show the GUI of the gym environment

map_desc = generate_random_map(size=BOARD_SZ, seed = FIXED_SEED)
env = gym.make('FrozenLake-v1', desc=map_desc, map_name=MAP_NAME, is_slippery=True, render_mode='human' if RENDER else None)
env = TimeLimit(env, TIME_LIMIT*BOARD_SZ)
action_size = env.action_space.n

# Creating the DQN agent (with greedy policy, suited for evaluation)
agent = DQNAgent(action_size, epsilon=0.0, epsilon_min=0.0)

# Checking if weights from previous learning session exists
if os.path.exists('frozen_lake.h5'):
    print('Loading weights from previous learning session.')
    agent.load("frozen_lake.h5")
else:
    print('No weights found from previous learning session. Unable to proceed.')
    exit(-1)
return_history = []
time_history = []
rate_of_completion = 0.0
for episode in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state, _ = env.reset()
    state = np.array([state])
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    prev_action = -1
    for time in range(1, TIME_LIMIT + 1):
        # Render the environment for visualization
        if RENDER:
            env.render()
        # Select action
        action = agent.act(state)
        # Take action, observe reward and new state
        next_state, completed, terminated, truncated, _ = env.step(action)
        next_state = np.array([next_state])
        # Making reward engineering to keep compatibility with how training was done
        reward = reward_engineering(state, prev_action, action, completed, next_state, terminated, truncated)
        prev_action = action
        state = next_state
        # Accumulate reward
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if terminated or truncated:
            print("episode: {}/{}, time: {}, score: {:.6f}, epsilon: {:.3f}"
                  .format(episode, NUM_EPISODES, time, cumulative_reward, agent.epsilon))
            time_history.append(time)
            break
    return_history.append(cumulative_reward)
    rate_of_completion += (completed - rate_of_completion)/episode
# Prints mean return
print('Mean return: ', np.mean(return_history))
print('Mean time: ', np.mean(time_history))
print('Rate of completion: {:.2f}%'.format(rate_of_completion*100))


# Plots return history
plt.plot(return_history, 'b')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('dqn_evaluation.eps', format='eps')