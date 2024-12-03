import csv
import os 
from Minimax import findBestMove

# Generador de datos
def data_generator(board = [0]*9, boards = [], i = 0):
    if i < 9:
        for value in [1, 0, -1]:  # Posibles valores
            new_board = board.copy()
            new_board[i] = value
            data_generator(new_board, boards, i + 1)
    elif (sum(board) == -1 and 0 in board):
        boards.append(board)  # Solo añadimos al final de la generación
    return boards

input_data = data_generator()

input_data.append([0]*9)

output_data = []

# Comprobamos que la salida y la entrada coincidan
for data in input_data:
    temp_output = findBestMove(data)
    output_data.append(temp_output)

# Get the directory path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, 'data/input_data.csv'), 'w', newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerows(input_data)

with open(os.path.join(script_dir, 'data/output_data.csv'), 'w', newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerows(output_data)
