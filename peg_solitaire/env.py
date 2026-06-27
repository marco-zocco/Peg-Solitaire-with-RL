

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .board import ENGLISH_MASK
from .moves import generate_move_table, legal_actions, apply_move

  
class PegSolitaireEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, mask=ENGLISH_MASK, start_state=None, render_mode=None, max_steps=200):
        super().__init__()
        self.mask = np.asarray(mask, dtype=bool)
        self.move_table = generate_move_table(self.mask) # set of all possible moves (w/o considering the state)
        self.render_mode = render_mode
        self._start_state = start_state  # None | array | callable(mask, rng) -> board
        self.max_steps = max_steps  # truncation cap: safety net for no-op illegal moves
        self._steps = 0             # steps taken in the current episode

        self.action_space = spaces.Discrete(len(self.move_table))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2, *self.mask.shape), dtype=np.float32
        )
        self.board = None

    # ---- start-state distribution: EXTENSION POINT (curriculum lives here) -------
    def _initial_board(self):
        """Default: canonical central-empty start (the position you report solving).
        """
        if callable(self._start_state):
            # custom start function
            return np.asarray(self._start_state(self.mask, self.np_random), dtype=bool)
        if self._start_state is not None:
            # fixed starting board
            return np.asarray(self._start_state, dtype=bool)
        #default
        board = self.mask.copy()
        cr, cc = self.mask.shape[0] // 2, self.mask.shape[1] // 2
        board[cr, cc] = False
        return board

    # ---- observation / legality -------------------------------------------------
    def _obs(self):
        peg = self.board.astype(np.float32)
        invalid = (~self.mask).astype(np.float32)
        return np.stack([peg, invalid], axis=0) 

    def action_mask(self):
        m = np.zeros(self.action_space.n, dtype=bool)
        m[legal_actions(self.board, self.move_table)] = True
        return m # same indexes for the moves as in move_table

    def successors(self):
        """
        outputs: (action_id, next_board) for each legal move.
        """
        return [
            (i, apply_move(self.board, self.move_table[i]))
            for i in legal_actions(self.board, self.move_table)
        ]

    # ---- Gymnasium API ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = self._initial_board()
        self._steps = 0
        return self._obs(), {"action_mask": self.action_mask()}

    def step(self, action):
        self._steps += 1

        # Illegal action -> no-op: board unchanged, no reward, not terminated.
        # truncated still fires if we've hit the step cap (next block).
        if action not in legal_actions(self.board, self.move_table):
            truncated = self._steps >= self.max_steps
            return self._obs(), 0.0, False, truncated, {"action_mask": self.action_mask()}

        self.board = apply_move(self.board, self.move_table[action])

        pegs = int(self.board.sum())
        won = pegs == 1
        stuck = not won and not legal_actions(self.board, self.move_table)
        terminated = won or stuck
        truncated = not terminated and self._steps >= self.max_steps  # safety net
        reward = 1.0 if won else 0.0  # sparse: the win is the only signal
        return self._obs(), reward, terminated, truncated, {"action_mask": self.action_mask()}

    def render(self):
        glyph = {(True, True): "o", (True, False): ".", (False, False): " "}
        lines = []
        for r in range(self.mask.shape[0]):
            row = [glyph[(self.mask[r, c], bool(self.board[r, c]))] for c in range(self.mask.shape[1])]
            lines.append(" ".join(row))
        return "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke test: a random legal playout from the central start.
    env = PegSolitaireEnv()
    obs, info = env.reset(seed=0)
    done, steps = False, 0
    while not done:
        legal = np.flatnonzero(info["action_mask"])
        if len(legal) == 0:
            break
        a = env.np_random.choice(legal)
        obs, r, done, _, info = env.step(int(a))
        steps += 1
    print(env.render())
    print(f"\npegs left: {int(env.board.sum())}  steps: {steps}  reward: {r}")
