import os
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from utils import reward_engineering
import tensorflow as tf

def plot_points(point_list, style):
    x = []
    y = []
    for point in point_list:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, style)


NUM_EPISODES = 30  # Number of episodes used for evaluation
RENDER = True

tf.compat.v1.disable_eager_execution()

map_desc = generate_random_map(size=4, seed = 0)
env = gym.make('FrozenLake-v1', desc=map_desc, map_name="4x4", is_slippery=True, render_mode='human' if RENDER else None)
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
for episodes in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state, _ = env.reset()
    state = np.array([state])
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    for time in range(1, 100):
        # Render the environment for visualization
        if RENDER:
            env.render()
        # Select action
        action = agent.act(state)
        # Take action, observe reward and new state
        next_state, reward, terminated, _, _ = env.step(action)
        next_state = np.array([next_state])
        # Making reward engineering to keep compatibility with how training was done
        reward = reward_engineering(state, action, reward, next_state)
        state = next_state
        # Accumulate reward
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if terminated:
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                  .format(episodes, NUM_EPISODES, time, cumulative_reward, agent.epsilon))
            time_history.append(time)
            break
    return_history.append(cumulative_reward)

# Prints mean return
print('Mean return: ', np.mean(return_history))
print('Mean time: ', np.mean(time_history))


# Plots return history
plt.plot(return_history, 'b')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('dqn_evaluation.eps', format='eps')

# # Plots the greedy policy learned by DQN
# plt.figure()
# position = np.arange(-1.2, 0.5 + 0.025, 0.05)
# velocity = np.arange(-0.07, 0.07 + 0.0025, 0.005)
# push_left = []
# none = []
# push_right = []
# for j in range(len(position)):
#     for k in range(len(velocity)):
#         pos = position[j]
#         vel = velocity[k]
#         state = np.array([[pos, vel]])
#         action = agent.act(state)
#         if action == 0:
#             push_left.append(state[0])
#         elif action == 1:
#             none.append(state[0])
#         else:
#             push_right.append(state[0])
# plot_points(push_left, 'b.')
# plot_points(none, 'r.')
# plot_points(push_right, 'g.')
# plt.xlabel('Position')
# plt.ylabel('Velocity')
# plt.title('Agent Policy')
# plt.legend(['Left', 'None', 'Right'])
# plt.savefig('agent_decision.eps', format='eps')
# plt.show()