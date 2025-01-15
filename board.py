# board.py

import numpy as np
from random import randint
from colorama import Fore, Style, init as colorama_init

colorama_init()

size = 8
j1_str = "O"
j2_str = "X"
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
        print(f'{Fore.LIGHTRED_EX}   {size - i}{Style.RESET_ALL}', end=' ')
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
        print(f'{Fore.LIGHTBLUE_EX}{i + 1}{Style.RESET_ALL}', end=' ')
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
    
    # 2) Print them in "rowcol" notation (or "pass" if pass_move) along with the original move id
    
    """
    print("MOVES:")
    sorted_moves = sorted(L, key=lambda m: (-m // size, m % size))
    for m in sorted_moves:
        if m == pass_move:
            print("pass")
        else:
            i = m // size       # i in [0..7], 0=top row
            j = m % size       # j in [0..7], 0=left column
            row_label = f'{Fore.LIGHTRED_EX}{size - i}{Style.RESET_ALL}'  # Red for rows
            col_label = f'{Fore.LIGHTBLUE_EX}{j + 1}{Style.RESET_ALL}'    # Blue for columns
            print(f"  {row_label}{col_label} {Fore.LIGHTBLACK_EX}({m}){Style.RESET_ALL}")
    """        
    
    # 3) Get user input
    s = input('\nVotre coup (ligne colonne sans espace, ou "pass" pour passer):')
    
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

# ------------------------------------------------
# EVALUATION DE TABLEAU

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

# ------------------------------------------------
# MINIMAX

def minimax(board, player, depth):
    """
    Simple Minimax implementation without alpha-beta pruning.

    Args:
        board (numpy.ndarray): The current game board.
        player (int): The player (j1 or j2) whose move it is.
        depth (int): The maximum depth of the search tree.

    Returns:
        tuple: (score, move), where score is the evaluation score and move is the best move.
    """
    # Check if we are at a terminal state or at the maximum depth
    if terminal(board) or depth == 0:
        return evaluate(board), None

    best_score = float('-inf') if player == j1 else float('inf')
    best_move = None

    for move in legal_moves(board, player):
        # Generate the new board state after making the move
        new_board = play(board, move, player)
        # Recursive minimax call with flipped player
        score, _ = minimax(new_board, -player, depth - 1)
        if player == j1:
            if score > best_score:
                best_score = score
                best_move = move
        else:
            if score < best_score:
                best_score = score
                best_move = move

    return best_score, best_move

# Nouvelle fonction : Minimax avec table de transpositions
def minimax_tr(board, player, depth, transposition_table):
    """
    Minimax implementation with transposition table.

    Args:
        board (numpy.ndarray): The current game board.
        player (int): The player (j1 or j2) whose move it is.
        depth (int): The maximum depth of the search tree.
        transposition_table (dict): The transposition table to store evaluated positions.

    Returns:
        tuple: (score, move), where score is the evaluation score and move is the best move.
    """
    board_key = (tuple(board), player)  # Clé unique pour la position

    # Vérifie la table de transpositions
    if board_key in transposition_table:
        stored_depth, score, move = transposition_table[board_key]
        if stored_depth >= depth:
            return score, move

    if terminal(board) or depth == 0:
        return evaluate(board), None

    best_score = float('-inf') if player == j1 else float('inf')
    best_move = None

    for move in legal_moves(board, player):
        new_board = play(board, move, player)
        score, _ = minimax_tr(new_board, -player, depth - 1, transposition_table)

        if player == j1:  # Maximizing player
            if score > best_score:
                best_score = score
                best_move = move
        else:  # Minimizing player
            if score < best_score:
                best_score = score
                best_move = move

    # Sauvegarde dans la table de transpositions
    transposition_table[board_key] = (depth, best_score, best_move)
    return best_score, best_move

# ------------------------------------------------
# ALPHABETA

def alphabeta(board, player, depth, alpha, beta):
    """
    Minimax implementation with alpha-beta pruning.

    Args:
        board (numpy.ndarray): The current game board.
        player (int): The player (j1 or j2) whose move it is.
        depth (int): The maximum depth of the search tree.
        alpha (float): The alpha value for pruning.
        beta (float): The beta value for pruning.

    Returns:
        tuple: (score, move), where score is the evaluation score and move is the best move.
    """
    if terminal(board) or depth == 0:
        return evaluate(board), None

    best_move = None

    if player == j1:
        best_score = float('-inf')
        for move in legal_moves(board, player):
            new_board = play(board, move, player)
            score, _ = alphabeta(new_board, -player, depth - 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
    else:
        best_score = float('inf')
        for move in legal_moves(board, player):
            new_board = play(board, move, player)
            score, _ = alphabeta(new_board, -player, depth - 1, alpha, beta)
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if beta <= alpha:
                break

    return best_score, best_move

def alphabeta_tr(board, player, depth, alpha, beta, transposition_table):
    """
    Alpha-Beta pruning implementation with transposition table and prioritized move exploration.

    Args:
        board (numpy.ndarray): The current game board.
        player (int): The player (j1 or j2) whose move it is.
        depth (int): The maximum depth of the search tree.
        alpha (float): The alpha value for pruning.
        beta (float): The beta value for pruning.
        transposition_table (dict): The transposition table to store evaluated positions.

    Returns:
        tuple: (score, move), where score is the evaluation score and move is the best move.
    """
    board_key = (tuple(board), player)

    # Check transposition table
    best_move = None
    if board_key in transposition_table:
        stored_depth, stored_score, stored_move = transposition_table[board_key]
        if stored_depth >= depth:
            return stored_score, stored_move
        # Use stored_move for prioritized exploration
        best_move = stored_move

    if terminal(board) or depth == 0:
        return evaluate(board), None

    legal_moves_list = legal_moves(board, player)
    if best_move in legal_moves_list:
        legal_moves_list.remove(best_move)
        legal_moves_list.insert(0, best_move)  # Prioritize stored best_move

    if player == j1:
        best_score = float('-inf')
        for move in legal_moves_list:
            new_board = play(board, move, player)
            score, _ = alphabeta_tr(new_board, -player, depth - 1, alpha, beta, transposition_table)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
    else:
        best_score = float('inf')
        for move in legal_moves_list:
            new_board = play(board, move, player)
            score, _ = alphabeta_tr(new_board, -player, depth - 1, alpha, beta, transposition_table)
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if beta <= alpha:
                break

    transposition_table[board_key] = (depth, best_score, best_move)
    return best_score, best_move

def alphabeta_itd(board, player, max_depth, transposition_table):
    """
    Iterative deepening for Alpha-Beta pruning with a transposition table.

    Args:
        board (numpy.ndarray): The current game board.
        player (int): The player (j1 or j2) whose move it is.
        max_depth (int): The maximum depth to explore.
        transposition_table (dict): The transposition table to store evaluated positions.

    Returns:
        int: The best move found.
    """
    best_move = None
    alpha = float('-inf')
    beta = float('inf')

    for depth in range(2, max_depth + 1):
        _, best_move = alphabeta_tr(board, player, depth, alpha, beta, transposition_table)

    return best_move

# ------------------------------------------------
# MONTE CARLO TREE SEARCH

def random_playout(board, player):
    """
    Performs a random playout from the current board position to the end of the game.

    Args:
        board (numpy.ndarray): Current game board.
        player (int): Current player.

    Returns:
        int: Reward for the current player (+1 for win, -1 for loss, 0 for draw).
    """
    current_board = np.copy(board)
    current_player = player

    while not terminal(current_board):
        moves = legal_moves(current_board, current_player)
        move = moves[randint(0, len(moves) - 1)]
        current_board = play(current_board, move, current_player)
        current_player = -current_player

    game_winner = winner(current_board)
    return 1 if game_winner == player else -1 if game_winner == -player else 0

class MCTS_Node:
    def __init__(self, parent, board, player, move=None):
        """
        Initialize a node for MCTS.

        Args:
            parent (MCTS_Node): Parent node.
            board (numpy.ndarray): Current game board.
            player (int): Player to move.
            move (int): Move leading to this node.
        """
        self.parent = parent
        self.board = copy_board(board)
        self.player = player
        self.move = move  # Move leading to this node
        self.legal_moves = legal_moves(board, player) if not terminal(board) else []
        self.children = []
        self.visits = 0
        self.total_reward = 0

    def mcts_run(self, max_iter=1000):
        for _ in range(max_iter):
            self.mcts_iteration()

    def mcts_iteration(self):
        node = self.select()
        if not terminal(node.board) and node.legal_moves:
            node = node.expand()
        reward = random_playout(node.board, node.player)
        node.backpropagate(reward)

    def select(self):
        """
        Selects the most promising child node using the UCB1 score.

        Returns:
            MCTS_Node: Selected child node.
        """
        current = self
        while current.children:
            current = max(current.children, key=lambda c: c.ucb_score())
        return current

    def expand(self):
        """
        Expands the current node by creating a child for one of the remaining legal moves.

        Returns:
            MCTS_Node: The newly created child node.
        """
        move = self.legal_moves.pop(randint(0, len(self.legal_moves) - 1))
        new_board = play(self.board, move, self.player)
        new_player = -self.player
        child = MCTS_Node(self, new_board, new_player, move)
        self.children.append(child)
        return child

    def backpropagate(self, reward):
        """
        Updates the node and propagates the reward up the tree.

        Args:
            reward (int): Reward for the current node's player.
        """
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            # Flip the reward for the parent (opponent perspective)
            self.parent.backpropagate(-reward)

    def ucb_score(self):
        """
        Calculates the Upper Confidence Bound for Trees (UCB1) score.

        Returns:
            float: UCB1 score.
        """
        if self.visits == 0:
            return float('inf')
        return (self.total_reward / self.visits +
                2 * (2 * np.log(self.parent.visits) / self.visits) ** 0.5)

    def best_move(self):
        """
        Returns the best move based on the number of visits.

        Returns:
            int: The best move.
        """
        if not self.children:
            return None
        best_child = max(self.children, key=lambda c: c.visits)
        return best_child.move

def ai_mcts(board, player, max_iter=1000):
    root = MCTS_Node(None, board, player)
    root.mcts_run(max_iter)
    return root.best_move()

# ------------------------------------------------
# EXPORTER

def ai_alea(board, player):
    L = legal_moves(board, player)
    x = randint(0, len(L) - 1)
    return L[x]

def ai_minimax(board, player):
    depth = 4
    _, move = minimax(board, player, depth)
    return move

def ai_minimax_tr(board, player, depth=4):
    transposition_table = {}
    _, move = minimax_tr(board, player, depth, transposition_table)
    return move

def ai_alphabeta(board, player, depth=4):
    alpha = float('-inf')
    beta = float('inf')
    _, move = alphabeta(board, player, depth, alpha, beta)
    return move

def ai_alphabeta_tr(board, player, depth=4):
    transposition_table = {}
    alpha = float('-inf')
    beta = float('inf')
    _, move = alphabeta_tr(board, player, depth, alpha, beta, transposition_table)
    return move

def ai_alphabeta_itd(board, player, max_depth=4):
    transposition_table = {}
    _, move = alphabeta_itd(board, player, max_depth, transposition_table)
    return move