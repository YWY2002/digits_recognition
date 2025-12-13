import numpy as np

def initParams(input_features, nodes):
    w1 = np.random.randn(nodes, input_features)
    b1 = np.random.randn(nodes)

    w2 = np.random.randn(nodes, nodes)
    b2 = np.random.randn(nodes)

    w3 = np.random.randn(10, nodes)
    b3 = np.random.randn(10)

    return w1, b1, w2, b2, w3, b3

def activateReLu(z):
    return np.maximum(0, z)

def one_hot_encoding(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y = one_hot_y[np.arange(y.size), y] = 1    #array[row, column]
    one_hot_y = one_hot_y.transpose()
    return one_hot_y

def softmax_column(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def forw_prop(w1, b1, w2, b2, w3, b3, x):
    z1 = w1.dot(x) + b1
    a1 = activateReLu(z1)

    z2 = w2.dot(a1) + b2
    a2 = activateReLu(z2)

    z3 = w3.dot(a2) + b3
    a3 = softmax_column(z3)

    return z1, a1, z2, a2, z3, a3

def back_prop(w1, b1, w2, b2, w3, b3, x, y):
    one_hot_y = one_hot_y(y)