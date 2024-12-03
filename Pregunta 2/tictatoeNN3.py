import random
import numpy as np

def print_board(board):
    """Prints the Tic-Tac-Toe board to the console."""
    for i in range(3):
        print("|", end="")
        for j in range(3):
            if board[i * 3 + j] == 0:
                print("   |", end="")
            elif board[i * 3 + j] == 1:
                print(" X |", end="")
            else:
                print(" O |", end="")
        print("\n-----------")

def is_winner(board, player):
    """Checks if the given player has won the game."""
    winning_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for condition in winning_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

def is_board_full(board):
    """Checks if the board is full."""
    return 0 not in board

def get_random_move(board):
    """Returns a random valid move from the available empty spaces."""
    empty_spaces = [i for i in range(9) if board[i] == 0]
    return random.choice(empty_spaces)

def get_best_move(board, player):
    """Returns the best move using a simple heuristic."""
    best_score = -float('inf')
    best_move = None
    for i in range(9):
        if board[i] == 0:
            board[i] = player
            score = -minimax(board, -player)
            board[i] = 0
            if score > best_score:
                best_score = score
                best_move = i
    return best_move

def minimax(board, player):
    """Minimax algorithm to evaluate the best move."""
    if is_winner(board, 1):
        return 1
    if is_winner(board, -1):
        return -1
    if is_board_full(board):
        return 0

    if player == 1:
        best_score = -float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = player
                score = -minimax(board, -player)
                board[i] = 0
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = player
                score = -minimax(board, -player)
                board[i] = 0
                best_score = min(score, best_score)
        return best_score

def play_game():
    """Plays a game of Tic-Tac-Toe."""
    board = [0] * 9
    player = 1  # Player 1 starts

    while True:
        print_board(board)

        if player == 1:
            move = get_best_move(board, player)
        else:
            move = get_random_move(board)

        board[move] = player

        if is_winner(board, player):
            print_board(board)
            print(f"Player {player} wins!")
            break
        elif is_board_full(board):
            print_board(board)
            print("It's a tie!")
            break

        player *= -1  # Switch players

if __name__ == "__main__":
    play_game()