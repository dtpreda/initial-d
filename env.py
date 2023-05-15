import gymnasium as gym

env = gym.make('CarRacing-v2', render_mode='human', continuous=False)

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation.shape)
    print(info)
    if terminated or truncated:
        observation = env.reset()

env.close()