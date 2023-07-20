import os
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from utils import reward_engineering, BOARD_SZ, TIME_LIMIT, MAP_NAME, FIXED_SEED, SLIPPERY, GENERAL_RP


if not os.path.exists('output/'):
    os.mkdir('output/')

# generates map from the seed
map_desc = generate_random_map(size=BOARD_SZ, seed = FIXED_SEED)
env = gym.make('FrozenLake-v1', desc=map_desc, map_name=MAP_NAME, is_slippery=SLIPPERY, max_episode_steps=TIME_LIMIT)
action_size = env.action_space.n
observation_sz = int(env.observation_space.n)
NUM_EPISODES = 2000  if GENERAL_RP else 500 # Number of episodes used for training

# Creating the DQN agent
agent = DQNAgent(observation_sz, action_size, buffer_size = 3*observation_sz)

batch_size = 4*BOARD_SZ  # batch size used for the experience replay
return_history = []

for episode in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state, _ = env.reset()
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    # previous time step action
    prev_action = -2
    f = 1.0
    for time in range(1, TIME_LIMIT + 1):
        # Select action
        action = agent.act(state)
        # Take action, observe reward and new state
        next_state, completed, terminated, truncated, _ = env.step(action)
        # Modifying reward
        reward = reward_engineering(state, prev_action, action, completed, next_state, terminated, truncated and not completed)
        prev_action = action
        # Appending this experience to the experience replay buffer
        agent.append_experience(state, action, reward, next_state)
        agent.update()
        state = next_state
        # Accumulate reward
        cumulative_reward += f*reward
        f *= agent.gamma
        if terminated or truncated:
            if episode % 50 == 0:
                print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                      .format(episode, NUM_EPISODES, time, cumulative_reward, agent.epsilon))
            break
        # We only update the policy if we already have enough experience in memory
        if len(agent.replay_buffer) > batch_size:
            agent.replay(batch_size)
    return_history.append(cumulative_reward)
    agent.update_epsilon()
    if episode % 50:
        #Saving the model to disk
        agent.save("output/frozen_lake.pkl")

agent.save("output/frozen_lake.pkl")
print("Greedy policy:")
print(agent.display_greedy_policy())

slip = '' if SLIPPERY else '_not_slippery'

plt.plot(return_history, 'b')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig("output/dqn_training_"+MAP_NAME+"_seed{}".format(FIXED_SEED)+slip+".eps", format='eps', bbox_inches='tight')