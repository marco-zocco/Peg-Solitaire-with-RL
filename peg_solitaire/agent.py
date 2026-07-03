"""

Target: y(s) = max over legal a of [ r + gamma * V_target(s') ]
  * Target network: V_target is a frozen copy used only to compute y; the live net
    is regressed toward it and synced every `target_sync` episodes. 
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
import os

from .env import PegSolitaireEnv
from .board import ENGLISH_MASK, featurize
from .moves import generate_move_table, legal_actions, apply_move, is_win, is_terminal
from .model import build_value_network
from .replay import ReplayBuffer
from .curriculum import generate_solvable_board
from .feasibility_solver import is_solvable


# ---- board <-> network input ------------------------------------------------
def featurize_batch(boards, mask):
    return np.stack([featurize(b, mask) for b in boards]).astype(np.float32)


def _v(model, X):
    """V(X) as a flat numpy array"""

    return np.asarray(model(X, training=False)).reshape(-1)



# ---- acting: greedy one-step lookahead --------------------------------------
def greedy_action(board, model, mask, move_table, legal):
    """argmax over legal moves of V_online(successor)"""

    succ = [apply_move(board, move_table[a]) for a in legal]
    vals = _v(model, featurize_batch(succ, mask))
    return legal[int(np.argmax(vals))]


def select_action(board, model, mask, move_table, legal, epsilon, rng):
    """"implementing epsilon-greedy"""

    if rng.random() < epsilon:
        return legal[rng.integers(len(legal))]
    return greedy_action(board, model, mask, move_table, legal)


# ---- fitted-Value Iteration target -------------------------------------------------------
def compute_targets(boards, terminals, target_model, mask, move_table, gamma):
    """
    input:
        boards: boards batch,
        terminals: (B,), terminal state per board,
        target_model: frozen target net,
        ...

    output:
        (B,) of the computed targets for each board.
    """

    B = len(boards)
    y = np.zeros(B, dtype=np.float32)

    succ_feats, owner, reward, bootstrap = [], [], [], []
    for i, board in enumerate(boards):
        if terminals[i]:
            continue  # target for terminal states is 0
        for a in legal_actions(board, move_table):
            new_b = apply_move(board, move_table[a])
            term = is_terminal(new_b, move_table)
            succ_feats.append(featurize(new_b, mask))
            owner.append(i)
            reward.append(1.0 if is_win(new_b) else 0.0)
            bootstrap.append(0.0 if term else 1.0)  # don't bootstrap past terminals

    if succ_feats:
        X = np.stack(succ_feats).astype(np.float32)
        v = _v(target_model, X)
        owner = np.asarray(owner)
        q = np.asarray(reward, np.float32) + gamma * np.asarray(bootstrap, np.float32) * v
        # for each board i, gather the Q-values of all its successor states (those whose owner == i) 
        # and write the maximum into y[i]
        for i in range(B):
            sel = owner == i
            if sel.any():
                y[i] = q[sel].max()
    return y


# ---- one gradient step ------------------------------------------------------
def td_update(model, optimizer, sampled_bords, targets):
    sampled_bords = tf.convert_to_tensor(sampled_bords)
    targets = tf.convert_to_tensor(targets)
    with tf.GradientTape() as tape:
        pred = model(sampled_bords, training=True)[:, 0]
        loss = tf.reduce_mean(tf.square(pred - targets))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss)


# ---- collection -------------------------------------------------------------
def random_rollout(board, move_table, buffer, rng, max_steps):
    """Fork from a branch point: play uniformly random to termination,
    storing every visited board. No model calls -> cheap CPU-wise."""
    n = 0
    for _ in range(max_steps):
        term = is_terminal(board, move_table)
        buffer.add(board, term)
        n += 1
        if term:
            break
        legal = legal_actions(board, move_table)
        a = legal[rng.integers(len(legal))]
        board = apply_move(board, move_table[a])
    return n


def collect_episode(model, mask, move_table, buffer, rng, depth, epsilon,
                    max_steps, branch_prob=0.0):
    board, _ = generate_solvable_board(mask, move_table, depth, rng)
    won = False
    main_adds = 0
    branch_adds = 0
    for _ in range(max_steps):
        term = is_terminal(board, move_table)
        buffer.add(board, term)
        main_adds += 1
        if term:
            won = is_win(board)
            break
        if branch_prob > 0.0 and rng.random() < branch_prob:
            branch_adds += random_rollout(board, move_table, buffer, rng, max_steps)
        legal = legal_actions(board, move_table)
        a = select_action(board, model, mask, move_table, legal, epsilon, rng)
        board = apply_move(board, move_table[a])
    return won, main_adds, branch_adds



# ---- schedules --------------------------------------------------------------
def linear_schedule(start, end, frac):
    
    return start + (end - start) * min(max(frac, 0.0), 1.0)

# ----- helpers ---------------------------------------------------------------
def _save_weights_atomic(net, path):
    tmp = path.replace(".weights.h5", ".tmp.weights.h5")
    net.save_weights(tmp)
    os.replace(tmp, path)


def _save_optimizer(optimizer, path):
    arrs = [np.asarray(v) for v in optimizer.variables]
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, *arrs)
    os.replace(tmp, path)


def _load_optimizer(optimizer, model, path):
    optimizer.build(model.trainable_variables)
    d = np.load(path)
    for v, k in zip(optimizer.variables, d.files):
        v.assign(d[k])


# ---- training loop ----------------------------------------------------------
def train(
    mask=ENGLISH_MASK,
    episodes=2000,
    gamma=0.98,
    lr=1e-3,
    buffer_capacity=50_000,
    batch_size=64,
    updates_per_episode=4,
    target_sync=20,            # episodes between hard syncs
    eps_start=1.0, eps_end=0.05, eps_anneal_frac=0.8,
    depth_start=2, depth_end=31, depth_anneal_frac=0.8,
    max_steps=200,
    seed=0,
    log_every=100,
    branch_prob=0.0,           # 0.0 -> baseline arm, >0 -> branched arm
    outdir=None,               # checkpoints/weights dir; enables auto-resume
    checkpoint_every=200,
):
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    env = PegSolitaireEnv(mask=mask, max_steps=max_steps)
    env.reset(seed=seed)
    move_table = env.move_table

    model = build_value_network(board_shape=mask.shape)
    target = build_value_network(board_shape=mask.shape)
    target.set_weights(model.get_weights())
    optimizer = Adam(lr)
    buffer = ReplayBuffer(buffer_capacity, board_shape=mask.shape)

    # --- solver-labeled probe set: live feasibility separation, guarantees both classes ---
    probe_rng = np.random.default_rng(seed + 12345)   # separate stream; don't perturb training RNG
    probe_boards, probe_labels, _probe_memo = [], [], {}
    while sum(probe_labels) < 20 or (len(probe_labels) - sum(probe_labels)) < 20:
        d = int(probe_rng.integers(6, 28))
        b, _ = generate_solvable_board(env.mask, env.move_table, d, probe_rng)
        for _ in range(int(probe_rng.integers(0, 6))):   # perturb some into dead territory
            la = legal_actions(b, env.move_table)
            if not la:
                break
            b = apply_move(b, env.move_table[int(probe_rng.choice(la))])
        probe_boards.append(b.copy())
        probe_labels.append(bool(is_solvable(b, env.move_table, _probe_memo)))
    probe_X = featurize_batch(probe_boards, env.mask)
    probe_labels = np.array(probe_labels)
    best_gap = -1.0

    # --- resume ---------------------------------------------------------
    start_ep = 0
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        spath = os.path.join(outdir, "state.json")
        if os.path.exists(spath):
            with open(spath) as f:
                st = json.load(f)
            start_ep = st["episode"] + 1
            best_gap = st["best_gap"]
            if start_ep >= episodes:
                print(f"[resume] {outdir} already complete, nothing to do", flush=True)
                model.load_weights(os.path.join(outdir, "final.weights.h5"))
                return model
            model.load_weights(os.path.join(outdir, "ckpt_model.weights.h5"))
            target.load_weights(os.path.join(outdir, "ckpt_target.weights.h5"))
            _load_optimizer(optimizer, model, os.path.join(outdir, "ckpt_optim.npz"))
            buffer.load(os.path.join(outdir, "ckpt_buffer.npz"))
            rng = np.random.default_rng(seed + 100_000 * start_ep)
            print(f"[resume] episode {start_ep} | buffer {len(buffer)} | best_gap {best_gap:+.2f}", flush=True)

    wins = 0
    main_adds = branch_adds = 0
    for ep in range(start_ep, episodes):
        frac = ep / max(episodes - 1, 1)
        epsilon = linear_schedule(eps_start, eps_end, frac / eps_anneal_frac)
        ceil  = int(round(linear_schedule(depth_start, depth_end, frac / depth_anneal_frac)))
        depth = int(rng.integers(2, ceil + 1))   # uniform [2, ceil]: endgame retained, mid-game still covered

        won, m, b = collect_episode(model, mask, move_table, buffer, rng,
                                    depth, epsilon, max_steps, branch_prob)
        wins += int(won)
        main_adds += m
        branch_adds += b

        loss = None
        if len(buffer) >= batch_size:
            for _ in range(updates_per_episode):
                boards, terms = buffer.sample(batch_size, rng)
                y = compute_targets(boards, terms, target, mask, move_table, gamma)
                X = featurize_batch(boards, mask)
                loss = td_update(model, optimizer, X, y)

        if ep % target_sync == 0:
            target.set_weights(model.get_weights())

        if outdir and ep > 0 and ep % checkpoint_every == 0:
            _save_weights_atomic(model, os.path.join(outdir, "ckpt_model.weights.h5"))
            _save_weights_atomic(target, os.path.join(outdir, "ckpt_target.weights.h5"))
            _save_optimizer(optimizer, os.path.join(outdir, "ckpt_optim.npz"))
            buffer.save(os.path.join(outdir, "ckpt_buffer.npz"))
            tmp = os.path.join(outdir, "state.json.tmp")
            with open(tmp, "w") as f:
                json.dump({"episode": ep, "best_gap": best_gap}, f)
            os.replace(tmp, os.path.join(outdir, "state.json"))

        if log_every and ep % log_every == 0:
            msg = f"ep {ep:5d} | depth {depth:2d} | eps {epsilon:.2f} | buffer {len(buffer):6d}"
            msg += f" | win-rate {wins / log_every:.2f}"
            tot = main_adds + branch_adds
            msg += f" | branch-frac {branch_adds / max(tot, 1):.2f}"
            if loss is not None:
                msg += f" | loss {loss:.4f}"
            pv  = _v(model, probe_X)
            gap = float(pv[probe_labels].mean() - pv[~probe_labels].mean())
            msg += f" | Vsolv {pv[probe_labels].mean():.2f} Vdead {pv[~probe_labels].mean():.2f} gap {gap:+.2f}"
            if gap > best_gap:
                best_gap = gap
                if outdir:
                    _save_weights_atomic(model, os.path.join(outdir, "best.weights.h5"))
            print(msg, flush=True)
            wins = 0
            main_adds = branch_adds = 0

    if outdir:
        _save_weights_atomic(model, os.path.join(outdir, "final.weights.h5"))
        tmp = os.path.join(outdir, "state.json.tmp")
        with open(tmp, "w") as f:
            json.dump({"episode": episodes - 1, "best_gap": best_gap}, f)
        os.replace(tmp, os.path.join(outdir, "state.json"))
    return model

# ---- evaluation hook ----------------------------------
def feasibility_scores(boards, model, mask):
    
    return _v(model, featurize_batch(boards, mask))


def evaluate_solve(env, model):
    """Greedy (epsilon=0) rollout from the canonical central-empty start"""

    _, action_mask = env.reset() 
    steps = 0
    while not is_terminal(env.board, env.move_table):
        legal = np.flatnonzero(action_mask["action_mask"])
        a = greedy_action(env.board, model, env.mask, env.move_table, legal)   # deterministic
        obs, reward, terminated, truncated, action_mask = env.step(a)
        steps += 1
        if truncated:
            break
    return is_win(env.board), steps, int(env.board.sum())


# ---- used for small tests -------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=12000)
    p.add_argument("--branch-prob", type=float, default=0.0)
    p.add_argument("--outdir", required=True)
    p.add_argument("--checkpoint-every", type=int, default=200)
    a = p.parse_args()

    train(
        episodes=a.episodes,
        gamma=1.0,
        lr=1e-3,
        buffer_capacity=50_000,
        batch_size=128,
        updates_per_episode=12,
        target_sync=20,
        eps_start=1.0, eps_end=0.15, eps_anneal_frac=0.65,
        depth_start=2, depth_end=31, depth_anneal_frac=0.5,
        max_steps=200,
        seed=a.seed,
        log_every=100,
        branch_prob=a.branch_prob,
        outdir=a.outdir,
        checkpoint_every=a.checkpoint_every,
    )
    print("done ->", a.outdir, flush=True)