"""
A mask is a 7x7 boolean np array: 1 = a real hole on the board, 2 = off-board.
"""

import numpy as np


def _mask_from_rows(rows):
    return np.array([[c == "1" for c in row] for row in rows], dtype=bool)


ENGLISH_MASK = _mask_from_rows([
    "0011100",
    "0011100",
    "1111111",
    "1111111",
    "1111111",
    "0011100",
    "0011100",
])

EUROPEAN_MASK = _mask_from_rows([
    "0011100",
    "0111110",
    "1111111",
    "1111111",
    "1111111",
    "0111110",
    "0011100",
])

