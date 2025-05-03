import numpy as np

class GridWorld:
    def __init__(self, size=5, start=(0, 0), goal=(4, 4)):
        self.size = size
        self.start = start
        self.goal = goal
        self.state = start
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

    def reset(self):
        self.state = self.start
        return self._state_to_index(self.state)

    def _state_to_index(self, state):
        """Convert (row, col) to a single integer index."""
        return state[0] * self.size + state[1]

    def _index_to_state(self, index):
        """Convert integer index back to (row, col)."""
        return (index // self.size, index % self.size)

    def step(self, action):
        """Apply action and return next state, reward, done."""
        move = self.actions[action]
        next_state = (self.state[0] + move[0], self.state[1] + move[1])

        # Check for wall collisions
        next_state = (
            max(0, min(self.size - 1, next_state[0])),
            max(0, min(self.size - 1, next_state[1]))
        )

        self.state = next_state
        reward = 1 if self.state == self.goal else -0.1
        done = self.state == self.goal

        return self._state_to_index(self.state), reward, done
