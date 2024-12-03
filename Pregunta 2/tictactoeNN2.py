import random
import numpy as np
import math

# Funciones auxiliares
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
    asignando recompensas para cada resultado.
    """
    boards = []
    if is_winner(board, 1) or is_winner(board, -1) or 0 not in board:
        reward = [1 if is_winner(board, 1) else 0 for _ in range(9)]
        return [(board[:], reward)]
    
    for i in range(9):
        if board[i] == 0:
            board[i] = current_player
            boards.extend(generate_boards(board, next_player, current_player))
            board[i] = 0  # Restaurar estado previo
    print(len(boards))
    return boards

def data_generator():
    """Genera los datos de entrenamiento iniciales."""
    initial_board = [0] * 9
    return generate_boards(initial_board, 1, -1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Clase de la red neuronal
class TicTacToeNN:
    def __init__(self):
        # Inicializar pesos y sesgos aleatoriamente
        self.input_to_hidden_weights = np.random.uniform(-1, 1, (9, 9))
        self.hidden_to_output_weights = np.random.uniform(-1, 1, (9, 9))
        self.hidden_biases = np.random.uniform(-1, 1, 9)
        self.output_biases = np.random.uniform(-1, 1, 9)

    def forward(self, inputs):
        """Propagación hacia adelante."""
        inputs = np.array(inputs)
        self.hidden_layer = sigmoid(np.dot(self.input_to_hidden_weights, inputs) + self.hidden_biases)
        self.output_layer = sigmoid(np.dot(self.hidden_to_output_weights, self.hidden_layer) + self.output_biases)
        return self.output_layer

    def backward(self, inputs, expected_output, learning_rate):
        """Propagación hacia atrás y actualización de pesos."""
        inputs = np.array(inputs)
        expected_output = np.array(expected_output)

        # Errores y deltas en la capa de salida
        output_errors = expected_output - self.output_layer
        output_deltas = output_errors * sigmoid_derivative(self.output_layer)

        # Errores y deltas en la capa oculta
        hidden_errors = np.dot(self.hidden_to_output_weights.T, output_deltas)
        hidden_deltas = hidden_errors * sigmoid_derivative(self.hidden_layer)

        # Actualizar pesos y sesgos
        self.hidden_to_output_weights += learning_rate * np.outer(output_deltas, self.hidden_layer)
        self.output_biases += learning_rate * output_deltas

        self.input_to_hidden_weights += learning_rate * np.outer(hidden_deltas, inputs)
        self.hidden_biases += learning_rate * hidden_deltas

    def train(self, training_data, epochs, learning_rate):
        """Entrena la red neuronal."""
        for epoch in range(epochs):
            total_error = 0
            for inputs, expected_output in training_data:
                self.forward(inputs)
                self.backward(inputs, expected_output, learning_rate)
                total_error += np.sum((expected_output - self.output_layer) ** 2)
            print(f"Epoch {epoch + 1}, Error: {total_error:.4f}")

if __name__ == '__main__':
    # Inicializa y entrena la red neuronal
    nn = TicTacToeNN()
    training_data = data_generator()
    nn.train(training_data, epochs=10, learning_rate=0.4)

    # Prueba la red neuronal
    test_input = [1, 0, 0, 0, -1, 0, 0, 0, 0]
    output = nn.forward(test_input)
    print("Output:", output)
