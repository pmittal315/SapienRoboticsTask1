import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("CartPole-v1")

# Initialize PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1
)

# Train the agent
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("ppo_cartpole")

# Evaluate performance
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=10, deterministic=True
)

print(f"PPO Mean Reward: {mean_reward}")
print(f"PPO Std Reward: {std_reward}")

env.close()
