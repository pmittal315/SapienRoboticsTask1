import gymnasium as gym
import time
from stable_baselines3 import DQN

# Create environment
env = gym.make("CartPole-v1", render_mode="human")

# Load trained SB3 DQN model
model = DQN.load("dqn_cartpole", env=env)

# Reset environment
obs, info = env.reset()

start_time = time.time()

# Run for 2 minutes
while time.time() - start_time < 120:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

