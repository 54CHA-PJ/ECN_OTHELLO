# tp-othello.py

from timeit import default_timer as timer

from board import init_board, print_board, play, terminal, j2, j1, winner, human, alea, j1_str, j2_str

b = init_board()
print_board(b)
player = j1

while not terminal(b):
    if player == j1:
        start = timer()
        m = human(b, player)
        end = timer()
        print(f'humain a joué en {end - start:.2f}s')
    else:
        start = timer()
        m = alea(b, player)
        end = timer()
        print(f'IA aléatoire a joué en {end - start:.2f}s')

    b = play(b, m, player)
    print_board(b)
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
