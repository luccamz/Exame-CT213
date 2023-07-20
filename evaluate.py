import os
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from utils import reward_engineering, BOARD_SZ, TIME_LIMIT, MAP_NAME, FIXED_SEED, SLIPPERY
import matplotlib.pyplot as plt
import seaborn as sns

NUM_EPISODES = 1000  # Number of episodes used for evaluation
RENDER = False # choose whether to show the GUI of the gym environment
if RENDER:
    NUM_EPISODES = 5

map_desc = generate_random_map(size=BOARD_SZ, seed = FIXED_SEED)
env = gym.make('FrozenLake-v1', desc=map_desc, map_name=MAP_NAME, is_slippery=SLIPPERY, render_mode='human' if RENDER else 'rgb_array', max_episode_steps=TIME_LIMIT)
action_size = env.action_space.n
observation_sz = env.observation_space.n

# Creating the DQN agent (with greedy policy, suited for evaluation)
agent = DQNAgent(observation_sz, action_size, epsilon=0.0, epsilon_min=0.0)

# Checking if weights from previous learning session exists
if os.path.exists('output/frozen_lake.pkl'):
    print('Loading Q table from previous learning session.')
    agent.load("output/frozen_lake.pkl")
else:
    print('No Q table found from previous learning session. Unable to proceed.')
    exit(-1)

#print(agent.q) # for troubleshooting

slip = '' if SLIPPERY else '_not_slippery'
values = agent.q_max_vals()
sns.heatmap(
        values,
        annot=agent.display_greedy_policy(plot_mode=True),
        fmt="",
        cmap='crest',
        mask=values== 0.0,
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"}, # change fontsize for greater board sizes 
    )
plt.savefig("output/learned_greedy_"+MAP_NAME+"_seed{}".format(FIXED_SEED)+slip+".eps", bbox_inches='tight')

return_history = []
time_history = []
rate_of_completion = 0.0
rate_of_loss_on_time = 0.0
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
        reward = reward_engineering(state, prev_action, action, completed, next_state, terminated, truncated and not completed)
        prev_action = action
        state = next_state
        # Accumulate reward
        cumulative_reward += f*reward
        f *= agent.gamma
        if terminated or truncated:
            if episode % 10 == 0:
                print("episode: {}/{}, time: {}, score: {:.6f}"
                      .format(episode, NUM_EPISODES, time, cumulative_reward))
            time_history.append(time)
            break
    return_history.append(cumulative_reward)
    rate_of_loss_on_time += ((truncated and not completed)- rate_of_loss_on_time)/episode
    rate_of_completion += (completed - rate_of_completion)/episode

if not RENDER:
    plt.imsave("output/last_frame_"+MAP_NAME+"_seed{}".format(FIXED_SEED)+slip +".eps",env.render()) # saves the last frame of the last episode

# Prints mean return
print('Mean return: ', np.mean(return_history))
print('Mean time: ', np.mean(time_history))
print('Rate of completion: {:.2f}%'.format(rate_of_completion*100))
print('Rate of loss on time: {:.2f}%'.format(rate_of_loss_on_time*100))

# Plots return history
plt.figure()
plt.plot(return_history, 'b')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig("output/dqn_evaluation_"+MAP_NAME+"_seed{}".format(FIXED_SEED)+slip+".eps", format='eps', bbox_inches='tight')