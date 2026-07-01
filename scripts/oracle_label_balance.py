import numpy as np
from peg_solitaire.env import PegSolitaireEnv
from peg_solitaire.model import build_value_network
from peg_solitaire.replay import ReplayBuffer
from peg_solitaire.agent import collect_episode, feasibility_scores
from peg_solitaire.feasibility_solver import is_solvable

env = PegSolitaireEnv(); mask, mt = env.mask, env.move_table
model = build_value_network(board_shape=mask.shape); model.load_weights("peg_seed0.weights.h5")

# reachable boards from forward play; epsilon>0 injects blunders
buf = ReplayBuffer(50_000, board_shape=mask.shape); rng = np.random.default_rng(0)
for _ in range(500):
    collect_episode(env, model, buf, rng, depth=31, epsilon=0.1)

B = buf.boards[:buf._size]
pegs = B.reshape(len(B), -1).sum(1) # flatten
V = feasibility_scores(B, model, mask)
memo = {}
solv = np.array([is_solvable(b, mt, memo) for b in B])  
np.savez("oracle_cache.npz", B=B, pegs=pegs, V=V, solv=solv)    # save the results

for lo, hi in [(2,5),(6,10),(11,16),(17,24),(25,32)]:
    m = (pegs >= lo) & (pegs <= hi)
    print(f"pegs {lo:2d}-{hi:2d}: n={int(m.sum()):4d}  solvable={int(solv[m].sum()):4d}  dead={int((~solv[m]).sum()):4d}")






def auc(v_solvable, v_dead):
    # exactly the sentence above: fraction of (solvable, dead) pairs
    # where V scores the solvable board higher (ties count as half)
    higher = (v_solvable[:, None] >  v_dead[None, :]).mean()
    tie    = (v_solvable[:, None] == v_dead[None, :]).mean()
    return higher + 0.5 * tie


d = np.load("oracle_cache.npz")
B, pegs, V, solv = d["B"], d["pegs"], d["V"], d["solv"]
for lo, hi in [(2, 5), (6, 10), (11, 16), (17, 24)]:
    band = (pegs >= lo) & (pegs <= hi)
    v_solv = V[band &  solv]     # V-scores of the solvable boards in this band
    v_dead = V[band & ~solv]     # V-scores of the dead boards in this band
    if len(v_solv) == 0 or len(v_dead) == 0:
        print(f"pegs {lo:2d}-{hi:2d}: skip (need both kinds)")
    else:
        print(f"pegs {lo:2d}-{hi:2d}: AUC = {auc(v_solv, v_dead):.3f}"
              f"   (solvable={len(v_solv)}, dead={len(v_dead)})")