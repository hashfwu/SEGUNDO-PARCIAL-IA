# Definimos los valores del jugador (1) y el oponente (-1)
player = 1
opponent = -1

# Esta función comprueba si hay movimientos disponibles en el tablero
def isMovesLeft(board):
    return 0 in board  # Si hay un 0, significa que hay espacio vacío

# Evaluación del estado del tablero
def evaluate(board):
    # Comprobar filas
    for row in range(3):
        if board[row * 3] == board[row * 3 + 1] == board[row * 3 + 2]:
            if board[row * 3] == player:
                return 10
            elif board[row * 3] == opponent:
                return -10

    # Comprobar columnas
    for col in range(3):
        if board[col] == board[col + 3] == board[col + 6]:
            if board[col] == player:
                return 10
            elif board[col] == opponent:
                return -10

    # Comprobar diagonales
    if board[0] == board[4] == board[8]:
        if board[0] == player:
            return 10
        elif board[0] == opponent:
            return -10
    if board[2] == board[4] == board[6]:
        if board[2] == player:
            return 10
        elif board[2] == opponent:
            return -10

    return 0  # No hay ganador aún

# Función Minimax para decidir el mejor movimiento
def minimax(board, depth, isMax):
    score = evaluate(board)

    # Si el jugador gana, devuelve +10
    if score == 10:
        return score

    # Si el oponente gana, devuelve -10
    if score == -10:
        return score

    # Si no hay movimientos, devuelve 0
    if not isMovesLeft(board):
        return 0

    if isMax:
        best = -1000
        # Buscar el mejor movimiento para el jugador
        for i in range(9):
            if board[i] == 0:  # Si el espacio está vacío
                board[i] = player
                best = max(best, minimax(board, depth + 1, not isMax))
                board[i] = 0
        return best
    else:
        best = 1000
        # Buscar el mejor movimiento para el oponente
        for i in range(9):
            if board[i] == 0:  # Si el espacio está vacío
                board[i] = opponent
                best = min(best, minimax(board, depth + 1, not isMax))
                board[i] = 0
        return best

# Función para encontrar el mejor movimiento
def findBestMove(board):
    bestVal = -1000
    bestMove = -1

    # Iterar a través de todos los posibles movimientos
    for i in range(9):
        if board[i] == 0:  # Si el espacio está vacío
            board[i] = player
            moveVal = minimax(board, 0, False)
            board[i] = 0

            if moveVal > bestVal:
                bestMove = i
                bestVal = moveVal

    # Realizar el movimiento en el tablero sin modificar el tablero original
    new_board = [0]*9
    new_board[bestMove] = player
    return new_board  # Devolver una copia del tablero con el mejor movimiento
