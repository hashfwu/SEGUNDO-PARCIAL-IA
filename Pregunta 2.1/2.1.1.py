import tensorflow as tf
import numpy as np

# Funciones auxiliares (proporcionadas por ti)
def is_winner(board, player):
    """Comprueba si un jugador ha ganado."""
    winning_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Filas
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columnas
        [0, 4, 8], [2, 4, 6]              # Diagonales
    ]
    return any(all(board[pos] == player for pos in line) for line in winning_positions)

def generate_boards(board, current_player, next_player):
    """
    Genera todos los posibles tableros desde un estado inicial,
    asignando recompensas para cada movimiento.
    """
    boards = []
    if is_winner(board, 1) or is_winner(board, -1) or 0 not in board:
        reward = [1 if board[i] == 0 and is_winner(board[:i] + [1] + board[i+1:], 1) else
                  -1 if board[i] == 0 and is_winner(board[:i] + [-1] + board[i+1:], -1) else
                  0 for i in range(9)]
        return [(board[:], reward)]
    
    for i in range(9):
        if board[i] == 0:
            board[i] = current_player
            boards.extend(generate_boards(board, next_player, current_player))
            board[i] = 0  # Restaurar estado previo
    return boards


def data_generator():
    """Genera los datos de entrenamiento iniciales."""
    initial_board = [0] * 9
    return generate_boards(initial_board, 1, -1)

# Generar datos
print("Generando datos de entrenamiento...")
data = data_generator()
print(f"Datos generados: {len(data)} ejemplos.")

# Convertir datos a formato NumPy
X = np.array([d[0] for d in data], dtype=np.float32)  # Tableros
y = np.array([d[1] for d in data], dtype=np.float32)  # Recompensas

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(9,)),       # Entrada: estado del tablero (9 casillas)
    tf.keras.layers.Dense(64, activation="relu"),  # Capa oculta
    tf.keras.layers.Dense(64, activation="relu"),  # Otra capa oculta
    tf.keras.layers.Dense(9, activation="linear")  # Salida: valores Q para cada casilla
])

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="mse",  # Error cuadrático medio
              metrics=["mae"])

# Entrenar el modelo
print("Entrenando el modelo...")
history = model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Probar el modelo
test_board = [0, 1, 0, -1, 1, -1, 0, 0, 0]  # Estado de prueba
prediction = model.predict(np.array([test_board]))
print("Predicción para el tablero de prueba:")
print(prediction)
