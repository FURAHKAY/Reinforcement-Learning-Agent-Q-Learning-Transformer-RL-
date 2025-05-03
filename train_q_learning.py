import numpy as np
from environments.gridworld import GridWorld
from agents.q_learning import QLearningAgent
import matplotlib.pyplot as plt

env = GridWorld(size=5)
agent = QLearningAgent(n_states=25, n_actions=4)

episodes = 500
rewards = []

for ep in range(episodes):
    s = env.reset()
    total_reward = 0
    done = False
    while not done:
        a = agent.act(s)
        s_prime, r, done = env.step(a)
        agent.update(s, a, r, s_prime)
        s = s_prime
        total_reward += r
    rewards.append(total_reward)

# Plot rewards
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-learning Reward Curve")
plt.savefig("plots/q_learning_reward_curve.png")
