"""
Start-state curriculum by REVERSE generation.

This only ever produces SOLVABLE boards.
Those enter the buffer the other way: the agent plays forward imperfectly from
these solvable starts and blunders into dead ends. 
"""


import numpy as np


def inverse_legal_actions(board, move_table):
    """
    output: Indices of moves whose INVERSE is playable
    """

    return [
        i for i, (s, m, d) in enumerate(move_table)
        if board[d] and not board[s] and not board[m]
    ]


def apply_inverse_move(board, move):
    s, m, d = move
    n_board = board.copy()
    n_board[d] = False
    n_board[m] = True
    n_board[s] = True
    return n_board


def generate_solvable_board(mask, move_table, depth, rng, win_cell=None):
    """A solvable board (depth = moves from a win)
    
    May stop early if no inverse move is available; returns the board
    and the depth actually reached.
    """

    valid = np.argwhere(mask) # playable board

    if win_cell is None:
        # if not specified the last peg cell is random
        win_cell = tuple(valid[rng.integers(len(valid))]) 

    board = np.zeros_like(mask)
    board[win_cell] = True

    reached = 0
    for _ in range(depth):
        inv = inverse_legal_actions(board, move_table)
        if not inv:
            break
        i = inv[rng.integers(len(inv))]
        board = apply_inverse_move(board, move_table[i])
        reached += 1
    return board, reached
