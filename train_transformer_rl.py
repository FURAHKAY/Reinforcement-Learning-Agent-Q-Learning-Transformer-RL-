import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from environments.gridworld import GridWorld
from agents.transformer_rl import TransformerRLAgent
import matplotlib.pyplot as plt

env = GridWorld(size=5)
model = TransformerRLAgent(state_dim=25, n_actions=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

rewards = []
episodes = 200
gamma = 0.99

for ep in range(episodes):
    s = env.reset()
    state_seq = [s]
    action_seq = []
    reward_seq = []
    done = False
    total_reward = 0

    while not done:
        # Pad or crop to fixed seq length (e.g. 10)
        seq_tensor = torch.tensor([state_seq[-10:]], dtype=torch.long)
        q_vals = model(seq_tensor)
        a = torch.argmax(q_vals).item()

        s_prime, r, done = env.step(a)

        # Estimate TD target
        seq_prime = torch.tensor([[*state_seq[-9:], s_prime]], dtype=torch.long)
        with torch.no_grad():
            q_next = model(seq_prime).max().item()

        td_target = r + gamma * q_next
        q_val = q_vals[0, a]

        # Optimize
        loss = loss_fn(q_val, torch.tensor(td_target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update history
        state_seq.append(s_prime)
        action_seq.append(a)
        reward_seq.append(r)
        total_reward += r
        s = s_prime

    rewards.append(total_reward)

# Plot
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Transformer RL Agent Reward Curve")
plt.savefig("plots/transformer_rl_reward_curve.png")
