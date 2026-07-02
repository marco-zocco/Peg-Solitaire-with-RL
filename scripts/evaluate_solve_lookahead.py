# Depth-d lookahead: root move = argmax over a bounded game-tree search,
# learned V used ONLY at the horizon

import numpy as np
from peg_solitaire.env import PegSolitaireEnv
from peg_solitaire.model import build_value_network
from peg_solitaire.moves import legal_actions, apply_move, is_win
from peg_solitaire.agent import featurize_batch, _v

WEIGHTS = "peg_seed0_best.weights.h5"

def make_V(model, mask):
    return lambda board: float(_v(model, featurize_batch([board], mask))[0])

def tree_value(board, d, mt, V, memo):
    """Best reachable outcome d plies ahead. Terminals -> truth; horizon -> V."""
    legal = legal_actions(board, mt)
    if not legal:                       # terminal
        return 1.0 if is_win(board) else 0.0
    if d == 0:                          # horizon: trust the learned value
        return V(board)
    key = (board.tobytes(), d)          # board is full state -> caching is exact
    if key in memo:
        return memo[key]
    best = max(tree_value(apply_move(board, mt[a]), d - 1, mt, V, memo) for a in legal)
    memo[key] = best
    return best

def search_action(board, depth, mt, V, memo):
    legal = legal_actions(board, mt)    # depth=1 == plain greedy
    vals = [tree_value(apply_move(board, mt[a]), depth - 1, mt, V, memo) for a in legal]
    return legal[int(np.argmax(vals))]

def evaluate_solve_lookahead(env, model, depth):
    V, memo = make_V(model, env.mask), {}   # memo valid all rollout (model fixed)
    _, am = env.reset(); steps = 0
    while True:
        legal = np.flatnonzero(am["action_mask"])
        if legal.size == 0:
            break
        a = search_action(env.board, depth, env.move_table, V, memo)
        _, _, term, trunc, am = env.step(a); steps += 1
        if term or trunc:
            break
    return bool(is_win(env.board)), steps, int(env.board.sum())

if __name__ == "__main__":
    env = PegSolitaireEnv()
    model = build_value_network(board_shape=env.mask.shape)
    model.load_weights(WEIGHTS)
    for d in (1, 2, 3, 4):
        r = evaluate_solve_lookahead(env, model, d)
        print(f"depth {d}: {r}   -> {'SOLVED' if r[0] else f'stuck@{r[2]}pegs'}")