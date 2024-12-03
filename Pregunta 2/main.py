
def init():
    print("Welcome to tic tac toe!\nWe will start with choosing teams.")
    one = input("Player 1, do you want to be O or X: ").upper()
    one = one if one in ("O", "X") else "O"
    two = "O" if one == "X" else "X"
    board = [[" " for _ in range(3)] for _ in range(3)]
    return one, two, board

def show(matrix, columns):
    print("   1   2   3 \n  ---+---+---")
    for column, row in zip(columns, matrix):
        print(f"{column}  {" | ".join(row)} \n  ---+---+---")

def winning(matrix):
    # Rows
    for row in matrix: # Row by row
        if all(c == row[0] and c != " " for c in row):
            return True
    # Columns
    rotated = [list(row) for row in zip(*matrix)] # Columns as rows
    rotated = [list(reversed(row)) for row in rotated] # Reverse to finish rotation
    for row in rotated:  # Row by row
        if all(c == row[0] and c != " " for c in row):
            return True
    # Diagonals
    for x, y in zip((0, 2), (2, 0)): # Left/right diagonals
        if matrix[0][x] == matrix[1][1] == matrix[2][y] != " ":
            return True

def draw(matrix):
    if all(c != " " for row in matrix for c in row):
        return True
    return False

def main():
    p1, p2, board = init()
    player = p1
    columns = 'a', 'b', 'c'
    while True:
        show(board, columns)
        other_player = p1 if player == p2 else p2
        if winning(board):
            print(f'Player {1 if other_player == p2 else 2}, you won!')
            break
        if draw(board):
            print('Its a draw!')
            break
        while True:
            cords = input(f"Player {1 if player == p1 else 2}, please enter the cords to fill (eg a2): ").lower()
            if len(cords) == 2 and cords[0] in columns and cords[1].isdigit() and 1 <= int(cords[1]) <= 3 and board[columns.index(cords[0])][int(cords[1]) - 1] == " ":
                board[columns.index(cords[0])][int(cords[1]) - 1] = player
                break
            print('Invalid cords.')
        player = other_player

if __name__ == "__main__":
    main()