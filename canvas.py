import numpy as np

def softmax(z):
    return np.exp(z) / sum(np.exp(z))

def derivReLu(z):
    return z > 0

matrix = np.array([[4, 2, 1],
                   [5, 6, 7]])

vector1 = np.array([3, 6, 3, 5, 6, 1, 4, 8, 9])
vector2 = np.array([2, 7, 2, 5, 6, 1, 4, 8, 9])
b1 = np.random.randn(16, 1)

correct = vector1 == vector2
accuracy = 1 / vector1.size * np.sum(correct)
print(accuracy)
# one_hot_Y = np.zeros((vector.size, vector.max() + 1))
# print(one_hot_Y)
# one_hot_Y[np.arange(vector.size), vector] = 1
# print(one_hot_Y)
# one_hot_Y = one_hot_Y.T
# print(one_hot_Y)