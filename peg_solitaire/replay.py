"""
Uniform experience replay over visited BOARDS (+ a terminal flag).
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, board_shape=(7, 7)):
        self.capacity = capacity
        self.boards = np.zeros((capacity, *board_shape), dtype=bool)
        self.terminal = np.zeros(capacity, dtype=bool)
        self._ptr = 0
        self._size = 0

    def add(self, board, terminal):
        self.boards[self._ptr] = board
        self.terminal[self._ptr] = terminal
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self._size, size=batch_size)
        return self.boards[idx].copy(), self.terminal[idx].copy()

    def __len__(self):
        return self._size
