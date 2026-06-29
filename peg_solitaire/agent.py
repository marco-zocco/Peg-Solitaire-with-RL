"""
Fitted-VI / TD agent: V + one-step lookahead, target network, uniform replay.

Target: y(s) = max over legal a of [ r + gamma * V_target(s') ]
  * Target network: V_target is a frozen copy used only to compute y; the live net
    is regressed toward it and synced every `target_sync` episodes. 
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam

from .board import ENGLISH_MASK
from .moves import generate_move_table, legal_actions, apply_move
from .model import build_value_network
from .replay import ReplayBuffer
from .curriculum import generate_solvable_board


# ---- board <-> network input ------------------------------------------------
def featurize(board, mask):
    """Board (7x7 bool) -> (7, 7, 2) float32: [peg plane, invalid plane]."""
    peg = board.astype(np.float32)
    invalid = (~mask).astype(np.float32)
    return np.stack([peg, invalid], axis=-1)  # channels-last for Keras


def featurize_batch(boards, mask):
    return np.stack([featurize(b, mask) for b in boards]).astype(np.float32)


def _v(model, X):
    """V(X) as a flat numpy array. Works for a Keras model or a numpy mock."""
    return np.asarray(model(X, training=False)).reshape(-1)


# ---- game predicates --------------------------------------------------------
def is_win(board):
    return int(board.sum()) == 1


def is_terminal(board, move_table):
    return is_win(board) or not legal_actions(board, move_table)


# ---- acting: greedy one-step lookahead --------------------------------------
def greedy_action(board, model, mask, move_table, legal):
    """argmax over legal moves of V_online(successor). Known dynamics -> no Q head."""
    succ = [apply_move(board, move_table[a]) for a in legal]
    vals = _v(model, featurize_batch(succ, mask))
    return legal[int(np.argmax(vals))]


def select_action(board, model, mask, move_table, legal, epsilon, rng):
    if rng.random() < epsilon:
        return legal[rng.integers(len(legal))]
    return greedy_action(board, model, mask, move_table, legal)


# ---- fitted-VI target -------------------------------------------------------
def compute_targets(boards, terminals, target_model, mask, move_table, gamma):
    """y for each sampled board. Terminal -> 0. Else max over successors of
    [r + gamma * (0 if successor terminal else V_target(successor))].

    All successors across the minibatch are batched into ONE forward pass through
    the target net, then reduced per board (segment max)."""
    B = len(boards)
    y = np.zeros(B, dtype=np.float32)

    succ_feats, owner, reward, bootstrap = [], [], [], []
    for i, board in enumerate(boards):
        if terminals[i]:
            continue  # value of being at an endpoint is 0
        for a in legal_actions(board, move_table):
            nb = apply_move(board, move_table[a])
            term = is_terminal(nb, move_table)
            succ_feats.append(featurize(nb, mask))
            owner.append(i)
            reward.append(1.0 if is_win(nb) else 0.0)
            bootstrap.append(0.0 if term else 1.0)  # don't bootstrap past terminals

    if succ_feats:
        X = np.stack(succ_feats).astype(np.float32)
        v = _v(target_model, X)
        owner = np.asarray(owner)
        q = np.asarray(reward, np.float32) + gamma * np.asarray(bootstrap, np.float32) * v
        for i in range(B):
            sel = owner == i
            if sel.any():
                y[i] = q[sel].max()
    return y


# ---- one gradient step ------------------------------------------------------
def td_update(model, optimizer, X, y):
    X = tf.convert_to_tensor(X)
    y = tf.convert_to_tensor(y)
    with tf.GradientTape() as tape:
        pred = model(X, training=True)[:, 0]
        loss = tf.reduce_mean(tf.square(pred - y))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss)


# ---- collection -------------------------------------------------------------
def collect_episode(model, mask, move_table, buffer, rng, depth, epsilon, max_steps):
    """Reverse-generate a solvable start, play forward epsilon-greedy, store every
    visited board (+ terminal flag). Returns whether this episode reached a win."""
    board, _ = generate_solvable_board(mask, move_table, depth, rng)
    won = False
    for _ in range(max_steps):
        term = is_terminal(board, move_table)
        buffer.add(board, term)
        if term:
            won = is_win(board)
            break
        legal = legal_actions(board, move_table)
        a = select_action(board, model, mask, move_table, legal, epsilon, rng)
        board = apply_move(board, move_table[a])
    return won


# ---- schedules --------------------------------------------------------------
def linear_schedule(start, end, frac):
    return start + (end - start) * min(max(frac, 0.0), 1.0)


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
    eps_start=1.0, eps_end=0.05, eps_anneal_frac=0.5,
    depth_start=2, depth_end=31, depth_anneal_frac=0.8,
    max_steps=200,
    seed=0,
    log_every=100,
):
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    move_table = generate_move_table(mask)
    model = build_value_network(board_shape=mask.shape)
    target = build_value_network(board_shape=mask.shape)
    target.set_weights(model.get_weights())
    optimizer = Adam(lr)
    buffer = ReplayBuffer(buffer_capacity, board_shape=mask.shape)

    wins = 0
    for ep in range(episodes):
        frac = ep / max(episodes - 1, 1)
        epsilon = linear_schedule(eps_start, eps_end, frac / eps_anneal_frac)
        depth = int(round(linear_schedule(depth_start, depth_end, frac / depth_anneal_frac)))

        won = collect_episode(model, mask, move_table, buffer, rng, depth, epsilon, max_steps)
        wins += int(won)

        loss = None
        if len(buffer) >= batch_size:
            for _ in range(updates_per_episode):
                boards, terms = buffer.sample(batch_size, rng)
                y = compute_targets(boards, terms, target, mask, move_table, gamma)
                X = featurize_batch(boards, mask)
                loss = td_update(model, optimizer, X, y)

        if ep % target_sync == 0:
            target.set_weights(model.get_weights())

        if log_every and ep % log_every == 0:
            msg = f"ep {ep:5d} | depth {depth:2d} | eps {epsilon:.2f} | buffer {len(buffer):6d}"
            msg += f" | win-rate {wins / log_every:.2f}"
            if loss is not None:
                msg += f" | loss {loss:.4f}"
            print(msg)
            wins = 0

    return model


# ---- evaluation hook (your oracle test lives here) --------------------------
def feasibility_scores(boards, model, mask):
    """V(board) for each board -> your feasibility score. Threshold > 0 to predict
    'solvable', then compare against YOUR backtracking solver's labels on a held-out
    set. The solver is consulted ONLY here, never during training."""
    return _v(model, featurize_batch(boards, mask))


if __name__ == "__main__":
    # Tiny smoke run: a few hundred shallow episodes just to prove the loop trains.
    # (Not a real run -- that needs the full schedule and many more episodes.)
    train(episodes=300, depth_start=2, depth_end=4, depth_anneal_frac=1.0,
          target_sync=10, log_every=50)
