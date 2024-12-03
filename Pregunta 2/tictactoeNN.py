import random
import math

def is_winner(board, player):
    # Define las condiciones para ganar
    winning_positions = [
        [0, 1, 2],  # Fila 1
        [3, 4, 5],  # Fila 2
        [6, 7, 8],  # Fila 3
        [0, 3, 6],  # Columna 1
        [1, 4, 7],  # Columna 2
        [2, 5, 8],  # Columna 3
        [0, 4, 8],  # Diagonal principal
        [2, 4, 6],  # Diagonal secundaria
    ]
    return any(all(board[pos] == player for pos in line) for line in winning_positions)

def generate_boards(board, current_player, next_player):
    boards = []
    if is_winner(board, 1) or is_winner(board, -1) or 0 not in board:
        return [(board[:], [1 if is_winner(board, 1) else 0 for _ in range(9)])]
    
    for i in range(9):
        if board[i] == 0:
            print(board)
            board[i] = current_player
            boards.extend(generate_boards(board, next_player, current_player))
            board[i] = 0  # Deshace el movimiento para explorar otros caminos
    
    return boards

def data_generator():
    initial_board = [0] * 9  # Tablero inicial vacío
    return generate_boards(initial_board, 1, -1)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class TicTacToeNN:
    def __init__(self):
        # Initialize weights and biases randomly
        self.input_to_hidden_weights = [[random.uniform(-1, 1) for _ in range(9)] for _ in range(9)]
        self.hidden_to_output_weights = [[random.uniform(-1, 1) for _ in range(9)] for _ in range(9)]
        self.hidden_biases = [random.uniform(-1, 1) for _ in range(9)]
        self.output_biases = [random.uniform(-1, 1) for _ in range(9)]

    def forward(self, inputs):
        # Calculate hidden layer values
        self.hidden_layer = []
        for i in range(9):
            activation = sum(self.input_to_hidden_weights[i][j] * inputs[j] for j in range(9)) + self.hidden_biases[i]
            self.hidden_layer.append(sigmoid(activation))

        # Calculate output layer values
        self.output_layer = []
        for i in range(9):
            activation = sum(self.hidden_to_output_weights[i][j] * self.hidden_layer[j] for j in range(9)) + self.output_biases[i]
            self.output_layer.append(sigmoid(activation))
        
        return self.output_layer

    def backward(self, inputs, expected_output, learning_rate):
        # Calculate output layer error and deltas
        output_errors = [expected_output[i] - self.output_layer[i] for i in range(9)]
        output_deltas = [output_errors[i] * sigmoid_derivative(self.output_layer[i]) for i in range(9)]

        # Calculate hidden layer error and deltas
        hidden_errors = [sum(output_deltas[j] * self.hidden_to_output_weights[j][i] for j in range(9)) for i in range(9)]
        hidden_deltas = [hidden_errors[i] * sigmoid_derivative(self.hidden_layer[i]) for i in range(9)]

        # Update hidden-to-output weights and biases
        for i in range(9):
            for j in range(9):
                self.hidden_to_output_weights[i][j] += learning_rate * output_deltas[i] * self.hidden_layer[j]
            self.output_biases[i] += learning_rate * output_deltas[i]

        # Update input-to-hidden weights and biases
        for i in range(9):
            for j in range(9):
                self.input_to_hidden_weights[i][j] += learning_rate * hidden_deltas[i] * inputs[j]
            self.hidden_biases[i] += learning_rate * hidden_deltas[i]

    def train(self, training_data, epochs, learning_rate):
        for epoch in range(epochs):
            total_error = 0
            for inputs, expected_output in training_data:
                self.forward(inputs)
                self.backward(inputs, expected_output, learning_rate)
                total_error += sum((expected_output[i] - self.output_layer[i]) ** 2 for i in range(9))
            print(f"Epoch {epoch + 1}, Error: {total_error}")


if __name__ == '__main__':
    # Test neural network
    nn = TicTacToeNN()
    training_data = data_generator()
    nn.train(training_data, 1, 0.4)
    test_input = [1, 0, 0, 0, -1, 0, 0, 0, 0]
    output = nn.forward(test_input)
    print("Output:", output)