import os
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.wrappers.time_limit import TimeLimit
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from utils import reward_engineering, BOARD_SZ, TIME_LIMIT, MAP_NAME, FIXED_SEED, SLIPPERY

NUM_EPISODES = 50  # Number of episodes used for evaluation
RENDER = False # choose whether to show the GUI of the gym environment

map_desc = generate_random_map(size=BOARD_SZ, seed = FIXED_SEED)
env = gym.make('FrozenLake-v1', desc=map_desc, map_name=MAP_NAME, is_slippery=SLIPPERY, render_mode='human' if RENDER else None)
env = TimeLimit(env, TIME_LIMIT)
action_size = env.action_space.n
observation_sz = env.observation_space.n

# Creating the DQN agent (with greedy policy, suited for evaluation)
agent = DQNAgent(observation_sz, action_size, epsilon=0.0, epsilon_min=0.0)

# Checking if weights from previous learning session exists
if os.path.exists('frozen_lake.pkl'):
    print('Loading weights from previous learning session.')
    agent.load("frozen_lake.pkl")
else:
    print('No weights found from previous learning session. Unable to proceed.')
    exit(-1)

print(agent.q) # for troubleshooting

# The policy as a table of the preferred action of each board slot
agent.display_greedy_policy()

return_history = []
time_history = []
rate_of_completion = 0.0
for episode in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state, _ = env.reset()
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    prev_action = -2
    f = 1
    for time in range(1, TIME_LIMIT + 1):
        # Render the environment for visualization
        if RENDER:
            env.render()
        # Select action
        action = agent.act(state)
        # Take action, observe reward and new state
        next_state, completed, terminated, truncated, _ = env.step(action)
        # Making reward engineering to keep compatibility with how training was done
        reward = reward_engineering(state, prev_action, action, completed, next_state, terminated, truncated)
        prev_action = action
        state = next_state
        # Accumulate reward
        cumulative_reward += f*reward
        f *= agent.gamma
        if terminated or truncated:
            print("episode: {}/{}, time: {}, score: {:.6f}"
                  .format(episode, NUM_EPISODES, time, cumulative_reward))
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