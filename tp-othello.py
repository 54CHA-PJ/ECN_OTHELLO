# tp-othello.py

from timeit import default_timer as timer

from board import init_board, print_board, play, terminal, j2, j1, winner, j1_str, j2_str

from board import evaluate, human, ai_alea, ai_minimax, ai_minimax_tr, ai_alphabeta, ai_alphabeta_tr, ai_mcts

# Fichier principal tp-othello.py avec intégration de ai_minmax_tr
b = init_board()
print_board(b)
player = j1

while not terminal(b):
    if player == j1:
        start = timer()
        m = ai_alea(b, player)
        end = timer()
        print(f'IA 1 (Aleatoire) a joué en {end - start:.2f}s')
    else:
        start = timer()
        m = ai_mcts(b, player, max_iter=10000)
        end = timer()
        print(f'IA 2 (MCTS) a joué en {end - start:.2f}s')

    b = play(b, m, player)
    print_board(b)
    print("Evaluation: ", evaluate(b))
    print()
    player = -player

print('Vainqueur: ', end='')
w = winner(b)
if w == j2:
    print(j2_str + " (j2)")
elif w == j1:
    print(j1_str + " (j1)")
else:
    print('égalité')