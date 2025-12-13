import numpy as np

def softmax(z):
    return np.exp(z) / sum(np.exp(z))

matrix = np.array([[4, 2, 1],
                   [5, 6, 7]])

vector = np.array([3, 7, 2, 5, 6, 1, 4, 8, 9])
zero = np.zeros((4, 5))


one_hot_Y = np.zeros((vector.size, vector.max() + 1))
print(one_hot_Y)
one_hot_Y[np.arange(vector.size), vector] = 1
print(one_hot_Y)
one_hot_Y = one_hot_Y.T
print(one_hot_Y)