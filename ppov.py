import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("ppo_cartpole")

env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
