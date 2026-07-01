# Ground-truth feasibility oracle


from .moves import legal_actions, apply_move, is_win

def is_solvable(board, move_table, memo=None):
    if is_win(board):                 
        return True
    if memo is None:
        memo = {}
    key = board.tobytes() 
    if key in memo:
        return memo[key]
    result = False
    for a in legal_actions(board, move_table):
        if is_solvable(apply_move(board, move_table[a]), move_table, memo):
            result = True
            break                           # one solution suffices
    memo[key] = result
    return result