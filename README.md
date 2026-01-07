# Task 1 – Reinforcement Learning on CartPole-v1 Environment

This repository contains **Task 1** implementation
The task demonstrates the use of the **CartPole-v1** environment and basic
implementations of **DQN (Deep Q-Network)** and **PPO (Proximal Policy Optimization)**.

---

## 📌 Task Objective

- Understand and interact with the CartPole reinforcement learning environment
- Demonstrate environment execution and visualization
- Implement DQN and PPO algorithms for CartPole
- Separate training and visualization logic for clarity

---

## 📂 File Description

### `task_1.py`
- Main execution file for Task 1
- Runs the CartPole-v1 environment using **random actions**
- Used to verify environment setup and rendering
- Displays the CartPole simulation window

---

### `dqn.py`
- Implementation of **Deep Q-Network (DQN)**
- Defines the neural network, replay buffer, and training loop
- Used for training an agent on the CartPole environment

---

### `dqnv.py`
- DQN variant with **visualization / evaluation**
- Loads or runs the DQN agent and displays its behavior in the environment

---

### `ppo.py`
- Implementation of **Proximal Policy Optimization (PPO)**
- Uses policy-gradient based reinforcement learning
- Handles policy updates and training logic for CartPole

---

### `ppov.py`
- PPO variant with **visualization / evaluation**
- Runs the PPO agent in the CartPole environment with rendering enabled

---

## ▶ How to Run

### 1️⃣ Install required libraries
```bash
pip install gymnasium stable-baselines3 torch numpy

### 1️⃣ Install required libraries
```bash


