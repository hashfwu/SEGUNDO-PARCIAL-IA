import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops import nn

# Configuración básica
board_size = 3
marks_win = 3
REWARD_WIN = 1.0
REWARD_DRAW = 0.0
REWARD_ACTION = 0.0
hidden_layer_size = 50
gamma = 0.8
epsilon_initial = 1.0
epsilon_final = 0.01
epsilon_anneal_episodes = 5000
learning_rate = 0.001
episode_max = 10000

# Funciones principales
def check_win(s):
    """Revisa filas, columnas y diagonales para determinar si hay un ganador."""
    for i in range(board_size):
        if all(s[i, :]) or all(s[:, i]):
            return True
    if all(s.diagonal()) or all(np.fliplr(s).diagonal()):
        return True
    return False

def apply_action(move_x, sx, so, a_index):
    """Aplica una acción al tablero y devuelve el nuevo estado."""
    (sx if move_x else so)[a_index] = True
    if check_win(sx if move_x else so):
        return REWARD_WIN, sx, so, True
    if np.all(sx + so):
        return REWARD_DRAW, sx, so, True
    return REWARD_ACTION, sx, so, False

class QNetwork(tf.keras.Model):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.flatten = layers.Flatten()
        self.hidden = layers.Dense(hidden_layer_size, activation='relu')
        self.output_layer = layers.Dense(board_size * board_size)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        x = self.output_layer(x)
        return tf.reshape(x, [-1, board_size, board_size])

def train():
    q_network = QNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    epsilon = epsilon_initial

    for episode_num in range(episode_max):
        sx, so = np.zeros((board_size, board_size), dtype=bool), np.zeros((board_size, board_size), dtype=bool)
        move_x = True

        while True:
            s_t = np.array([[sx, so]] if move_x else [[so, sx]], dtype=np.float32)
            q_t = q_network(s_t).numpy()[0]
            valid_moves = np.transpose(np.where((sx + so) == False))
            a_index = tuple(
                valid_moves[np.random.randint(len(valid_moves))] 
                if np.random.random() < epsilon else 
                np.unravel_index(np.argmax(q_t), q_t.shape)
            )
            r_t, sx, so, terminal = apply_action(move_x, sx, so, a_index)

            if terminal:
                break
            move_x = not move_x

        epsilon = max(epsilon_final, epsilon - (epsilon_initial - epsilon_final) / epsilon_anneal_episodes)

def main():
    train()

if __name__ == "__main__":
    main()
