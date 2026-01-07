import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("CartPole-v1")

# Initialize DQN model
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1
)

# Train the agent
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("dqn_cartpole")

# Evaluate performance
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=10, deterministic=True
)

print(f"DQN Mean Reward: {mean_reward}")
print(f"DQN Std Reward: {std_reward}")

env.close()
