import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.Q[state]))
        return np.argmax(self.Q[state])

    def update(self, s, a, r, s_prime):
        td_target = r + self.gamma * np.max(self.Q[s_prime])
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])
