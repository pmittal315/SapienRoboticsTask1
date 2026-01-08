import gymnasium as gym
import time

# Create environment
env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

start_time = time.time()

# Run continuously for 2 minutes
while time.time() - start_time < 120:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

