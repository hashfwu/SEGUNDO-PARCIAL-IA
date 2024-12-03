import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import os

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

class TicTacToeNN:
    def __init__(self, layer_sizes):
        """
        Inicializa la red neuronal.
        :param layer_sizes: Lista de tamaños de cada capa [input_size, hidden1_size, ..., output_size].
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        # Inicializar pesos y sesgos
        self.weights = [np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i + 1])) for i in range(self.num_layers)]
        self.biases = [np.random.uniform(-1, 1, (1, layer_sizes[i + 1])) for i in range(self.num_layers)]
        
        self.loss = []

    def train(self, input_data, output_data, epochs, learning_rate, view_rate=None, view_graph=False):
        if view_graph:
            fig, ax = plt.subplots()
        else:
            plt.clf()
            display.clear_output(wait=True)

        for epoch in range(epochs):
            # Propagación hacia adelante
            activations = [input_data]
            for i in range(self.num_layers):
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                a = tanh(z) if i < self.num_layers - 1 else 1 / (1 + np.exp(-z))  # Activación tanh excepto en la salida (sigmoid)
                activations.append(a)

            # Cálculo del error
            error = output_data - activations[-1]
            mse = np.mean(np.square(error))

            # Propagación hacia atrás
            deltas = [error * (activations[-1] * (1 - activations[-1]))]  # Derivada de sigmoid
            for i in range(self.num_layers - 1, 0, -1):
                delta = np.dot(deltas[0], self.weights[i].T) * tanh_derivative(activations[i])
                deltas.insert(0, delta)

            # Actualización de pesos y sesgos
            for i in range(self.num_layers):
                self.weights[i] += np.dot(activations[i].T, deltas[i]) * learning_rate
                self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate

            if view_rate and epoch % view_rate == 0:
                if view_graph:
                    self.loss.append(mse)
                    ax.clear()
                    ax.plot(self.loss, label="Pérdida (MSE)", color="blue")
                    ax.set_title(f"Epoch {epoch}/{epochs}    MSE: {mse}")
                    ax.set_xlabel(f"Épocas (cada {view_rate})")
                    ax.set_ylabel("Pérdida / Error")
                    ax.legend()
                    display.clear_output(wait=True)
                    display.display(plt.gcf())
                else:
                    print(f"Epoch {epoch}/{epochs} - MSE: {mse}")
        plt.clf()
        print(f"Error final (MSE): {mse}")

    def predict(self, input_data):
        activations = input_data
        for i in range(self.num_layers):
            z = np.dot(activations, self.weights[i]) + self.biases[i]
            activations = tanh(z) if i < self.num_layers - 1 else 1 / (1 + np.exp(-z))  # Activación
        return activations

    def save(self):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            np.save(f"weights_layer_{i}.npy", w)
            np.save(f"bias_layer_{i}.npy", b)

    # def load(self):
    #     self.weights = []
    #     self.biases = []
    #     for i in range(self.num_layers):
    #         if os.path.exists(f"weights_layer_{i}.npy") and os.path.exists(f"bias_layer_{i}.npy"):
    #             self.weights.append(np.load(f"weights_layer_{i}.npy"))
    #             self.biases.append(np.load(f"bias_layer_{i}.npy"))
    #         else:
    #             print(f"No se encontraron los archivos para la capa {i}.")
    def load(self):
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            weight_file = f"weights_layer_{i}.npy"
            bias_file = f"bias_layer_{i}.npy"

            if os.path.exists(weight_file) and os.path.exists(bias_file):
                # Cargar pesos y sesgos
                weights = np.load(weight_file)
                biases = np.load(bias_file)

                # Verificar formas
                expected_weight_shape = (self.layer_sizes[i], self.layer_sizes[i + 1])
                expected_bias_shape = (1, self.layer_sizes[i + 1])

                if weights.shape != expected_weight_shape:
                    raise ValueError(
                        f"Incompatibilidad en los pesos de la capa {i}.\n"
                        f"Esperado: {expected_weight_shape}, Encontrado: {weights.shape}\n"
                        f"Archivo: {weight_file}"
                    )

                if biases.shape != expected_bias_shape:
                    raise ValueError(
                        f"Incompatibilidad en los sesgos de la capa {i}.\n"
                        f"Esperado: {expected_bias_shape}, Encontrado: {biases.shape}\n"
                        f"Archivo: {bias_file}"
                    )

                # Añadir a la red si las formas son correctas
                self.weights.append(weights)
                self.biases.append(biases)
            else:
                raise FileNotFoundError(f"No se encontraron los archivos para la capa {i}:\n"
                                        f"Pesos: {weight_file}, Sesgos: {bias_file}")

        print("Pesos y sesgos cargados correctamente.")

