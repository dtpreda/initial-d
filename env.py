import gymnasium as gym
import torch
import random
import numpy as np
from reinforce import REINFORCE

env = gym.make('CarRacing-v2', render_mode='human')
env = gym.wrappers.RecordEpisodeStatistics(env, 50)
# print(env.observation_space.shape)
total_num_episodes = 1000
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
reward_list = []

observation, info = env.reset()

for seed in [1, 2, 3, 5, 8]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    agent = REINFORCE(action_dim)
    episode_reward_list = []

    for episode in range(total_num_episodes):
        print("Episode: ", episode)
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            action = agent.get_action(obs)
            action = action[0].tolist()
            obs, reward, terminated, truncated, info = env.step(action)

            agent.rewards.append(reward)
            done = terminated or truncated
        
        episode_reward_list.append(env.return_queue[-1])
        agent.update()

        if episode % 10 == 0:
            avg_reward = int(np.mean(env.return_queue))
            print(f"Episode: {episode}, Reward: {avg_reward}")

    reward_list.append(episode_reward_list)

env.close()