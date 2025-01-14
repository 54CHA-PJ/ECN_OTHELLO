# board.py

import numpy as np
from random import randint
from colorama import Fore, Style

size = 8

j1_str = "O"
j2_str = "X"
# j1_str = "\x1b[1m\x1b[31mO\x1b[0m"
# j2_str = "\x1b[1m\x1b[34mO\x1b[0m"

j1 = 1
j2 = -1
empty = 0

pass_move = size * size

# directions, dans le sens direct, en commençant par la droite
directions = (1, 1 - size, -size, -1 - size, -1, -1 + size, size, 1 + size)

# renvoie le signe de x
def sign(x):
    return int(x > 0) - int(x < 0)

# limites d'itération en fonction de la direction, dans le sens direct, en commençant par la droite
limits = [[(size - 1 - j, min(i, size - 1 - j), i, min(i, j), j, min(size - 1 - i, j), size - 1 - i, min(size - 1 - i, size - 1 - j))
            for j in range(size)] for i in range(size)]

# indique s'il y a des pions à retourner dans une direction donnée
# player est la couleur de celui qui joue
# limit est la limite à ne pas dépasser afin de ne pas sortir du plateau
def check1D(board, position, player, direction, limit):
    k = 1
    position += direction
    while k <= limit and board[position] == -player:
        position += direction
        k = k + 1

    # Il faut qu'il y ait eu au moins un pion adverse
    return (k > 1 and k <= limit and board[position] == player)

# Calcule le suivant dans une direction donnée 
# player est la couleur de celui qui joue
# limit est la limite à ne pas dépasser afin de ne pas sortir du plateau
# On suppose qu'on doit effectivement retourner des pions dans cette direction
# i. e. check1D a été appelé avant
def play1D(board, position, player, direction, limit):
    k = 1
    position += direction
    while k <= limit and board[position] == -player:
        board[position] = player
        position += direction
        k = k + 1

# Calcule le successeur en place, obtenu en ajoutant un pion de player
# sur la position donnée
def play_(board, position, player):
    if position != pass_move:
        # La position en 2D
        i = position // size   
        j = position % size

        # La case jouée
        board[position] = player

        # Retourne les pions dans toutes les directions
        for direction, limit in zip(directions, limits[i][j]):
            if check1D(board, position, player, direction, limit):
                play1D(board, position, player, direction, limit) 

# Successeur avec copie
def play(board, position, player):
    r = np.copy(board)
    play_(r, position, player)

    return r

# Intialise le plateau
def init_board():
    # Crée et initialise tout à vide
    b = np.array([empty for k in range(size*size)])

    # Place les quatre premiers pions
    b[3 * size + 3] = j2
    b[4 * size + 4] = j2
    b[3 * size + 4] = j1
    b[4 * size + 3] = j1

    return b

# copy for MCTS
def copy_board(board):
    return np.copy(board)

# get a key for transposition tables
def board_key(board):
    return tuple(board)

# Affiche le plateau
def print_board(board):
    print()
    for i in range(size):
        # Affiche le numéro de la ligne en rouge
        print(f'{Fore.RED}   {size - i}{Style.RESET_ALL}', end=' ')
        for j in range(size):
            # Contenu de la case
            p = board[i * size + j]

            if p == empty:
                print("·", end=' ')
            elif p == j1:
                print(j1_str, end=' ')
            elif p == j2:
                print(j2_str, end=' ')
            else:
                print("?", end=' ')
        print()
    # Numéros de colonne en bleu
    print("     ", end='')
    for i in range(size):
        print(f'{Fore.BLUE}{i + 1}{Style.RESET_ALL}', end=' ')
    print()
    print()

# Trouve les coups légaux pour le joueur player
def legal_moves(board, player):
    L = []
    for p in range(size * size):
        # Il faut au moins que la case soit vide
        if board[p] == empty:
            # On cherche au moins une direction dans laquelle c'est valide
            i = p // size
            j = p % size
            
            lims = limits[i][j]

            valid = False
            n = 0
            while n < 8 and not valid:
                valid = check1D(board, p, player, directions[n], lims[n]) 
                n = n + 1

            # Valide, on enregistre
            if valid:
                L.append(p)

    # Pas de coup à jouer: il faut passer
    if not L:
        L.append(pass_move)

    return L

# Vérifie si la partie est finie
# Similaire à legal_moves mais on teste les deux joueurs
# pour chaque case
def terminal(board):
    r = True
    p = 0
    while p < size * size and r:
        # Il faut au moins que la case soit vide
        if board[p] == empty:
            # On cherche au moins une direction dans laquelle c'est valide
            i = p // size
            j = p % size

            lims = limits[i][j]

            n = 0
            while n < 8 and r:
                # check1D nous dit si on peut jouer dans cette direction
                # si on peut alors la position n'est pas terminale
                r = not (check1D(board, p, j1, directions[n], lims[n])
                    or check1D(board, p, j2, directions[n], lims[n])) 
                n = n + 1

        p = p +1

    return r

# Trouve le joueur qui a le plus de pions
def winner(board):
    score = 0
    for i in range(size * size):
        score += board[i]

    return sign(score)

# Demande un coup à un joueur humain
# on attend soit pass quand passer est autorisé
# ou un coup du type 35 : ligne 3, colonne 5
def human(board, player):
    # 1) Collect legal moves
    L = legal_moves(board, player)
    
    # 2) Print them in "rowcol" notation (or "pass" if pass_move)
    print("MOVES:")
    for m in L:
        if m == pass_move:
            print("pass")
        else:
            i = m // size       # i in [0..7], 0=top row
            j = m % size       # j in [0..7], 0=left column
            row_label = f'{Fore.RED}{size - i}{Style.RESET_ALL}'  # Red for rows
            col_label = f'{Fore.BLUE}{j + 1}{Style.RESET_ALL}'    # Blue for columns
            print(f"{row_label}{col_label}")
    
    # 3) Get user input
    s = input('Votre coup (ligne colonne sans espace, ou "pass" pour passer):')
    
    # 4) Convert input to a move index
    if s == 'pass':
        m = pass_move
    else:
        # Make sure the user typed exactly 2 digits
        if len(s) != 2:
            print('Coup interdit')
            return human(board, player)
        try:
            x = int(s[0])  # row label from top
            y = int(s[1])  # col label
        except ValueError:
            print('Coup interdit')
            return human(board, player)
        
        # Also check they are in range 1..8
        if not (1 <= x <= 8 and 1 <= y <= 8):
            print('Coup interdit')
            return human(board, player)
        
        # Convert row-col back to 1D index
        # row_label = x => i = 8 - x
        # col_label = y => j = y - 1
        i = size - x
        j = y - 1
        m = i * size + j
    
    # 5) Validate the move is in L
    if m in L:
        return m
    else:
        print('Coup interdit')
        return human(board, player)

# Donne un coup légal aléatoire
def alea(board, player):
    L = legal_moves(board, player)
    x = randint(0, len(L) - 1)

    return L[x]

