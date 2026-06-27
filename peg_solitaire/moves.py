"""
Move geometry generated from the mask.

A move is one peg jumping orthogonally over an adjacent peg into the empty hole
two cells away. Represented as (start, mid, dest), each an (row, col) tuple.
"""

import numpy as np

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


def generate_move_table(mask):
    """
    Every geometrically-possible jump on this board SHAPE.

    Independent of where pegs currently sit.
    The list index is the action id, and that index is what the Gymnasium Discrete action space refers to.
    """
    moves = []
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            for d_r, d_c in DIRECTIONS:
                m_r, m_c = r + d_r, c + d_c
                dr2, dc2 = r + 2 * d_r, c + 2 * d_c
                if 0 <= dr2 < rows and 0 <= dc2 < cols and mask[m_r, m_c] and mask[dr2, dc2]:
                    moves.append(((r, c), (m_r, m_c), (dr2, dc2)))
    return moves


def legal_actions(board, move_table):
    """
    Indices of moves playable from this occupancy: src has a peg, mid has a peg,
    dst is empty.
    """
    return [
        i for i, (s, m, d) in enumerate(move_table)
        if board[s] and board[m] and not board[d]
    ]


def apply_move(board, move):
    """
    Return a NEW board with `move` applied
    """
    s, m, d = move
    new_board = board.copy()
    new_board[s] = False   # peg leaves source
    new_board[m] = False   # jumped peg is removed
    new_board[d] = True    # peg lands in destination
    return new_board
