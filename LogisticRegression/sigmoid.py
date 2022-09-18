import numpy as np

def sigmoid(matrix):
    return 1 / (1 + np.exp(-matrix))