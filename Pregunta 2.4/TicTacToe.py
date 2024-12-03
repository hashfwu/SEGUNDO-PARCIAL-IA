import numpy as np
from NeuralNetwork import TicTacToeNN

layer_sizes = [9, 27, 81, 243, 81, 27, 9]  # Define las capas
model = TicTacToeNN(layer_sizes)

model.load()
# model.predict([0,0,0,0,0,0,0,0,0])

# class Tablero:
#     def __init__(self, turno):
#         self.tablero = [0] * 9
#         self.turno = turno
    
#     def show(self):
#         for i in range(9):
#             if i%3 == 0:
#                 print('\n')
#             else:
#                 print(self.tablero[i], end=' ')

# def main():
#     print('Bienvenido al juengo TicTacToe con redes neuronales')
#     simbolo = input('Quiere ser X o O')
#     turno = input('1. Quiero empezar\n2. A ver que tal lo hace la red: ')
#     tablero = Tablero(turno)
#     tablero.show()
