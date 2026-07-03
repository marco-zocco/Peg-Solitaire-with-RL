"""
Uniform experience replay over visited BOARDS (+ a terminal flag).
"""

import numpy as np
import os


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
    

    def save(self, path):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, boards=self.boards, terminal=self.terminal,
                                ptr=self._ptr, size=self._size)
        os.replace(tmp, path)

    def load(self, path):
        d = np.load(path)
        self.boards = d["boards"]
        self.terminal = d["terminal"]
        self._ptr = int(d["ptr"])
        self._size = int(d["size"])
