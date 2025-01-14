
# valeur des cases pour l'évaluation statique
basic_vals = [
    10, -5,  3,  3,  3,  3, -5, 10,
    -5, -5, -1, -1, -1, -1, -5, -5,
     3, -1,  1,  1,  1,  1, -1,  3,
     3, -1,  1,  1,  1,  1, -1,  3,
     3, -1,  1,  1,  1,  1, -1,  3,
     3, -1,  1,  1,  1,  1, -1,  3,
    -5, -5, -1, -1, -1, -1, -5, -5,
    10, -5,  3,  3,  3,  3, -5, 10]

# Évaluation statique des positions
def evaluate(board):
    score = 0

    # Liberté de movement
    freedom = len(legal_moves(board, j1)) - len(legal_moves(board, j2))

    # Nombre de pions
    disks = 0
    total_disks = 0
    for i, p in enumerate(board):
        disks += p * basic_vals[i]
        total_disks += abs(p)

    if total_disks == size * size:
        score = winner(board) * 10000
    else:
        score = disks + freedom

    return score
