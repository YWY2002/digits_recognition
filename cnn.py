import numpy as np


#X[784, m] -> 1[16, m] -> 2[16, m] -> Output[10, m]

def initParams(input_features, nodes):
    w1 = np.random.randn(nodes, input_features) 
    b1 = np.random.randn(nodes, 1) 

    w2 = np.random.randn(nodes, nodes) 
    b2 = np.random.randn(nodes, 1) 

    w3 = np.random.randn(10, nodes) 
    b3 = np.random.randn(10, 1) 

    return w1, b1, w2, b2, w3, b3

def activateReLu(z):
    return np.maximum(0, z)

def derivReLu(z):
    return z > 0

def one_hot_encoding(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1    #array[row, column]
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

    #a1/a2[16, m]   a3[10, m]
    return z1, a1, z2, a2, z3, a3

'''
def back_prop(z1, w1, a1, z2, w2, a2, w3, a3, x, y):
    one_hot_y = one_hot_encoding(y)
    dZ3 = a3 - one_hot_y
    dW3 = a2.dot(dZ3.transpose())   #dW3[16, 10]
    db3 = a3 - one_hot_y            #db3[10, m]
    dZ2 = w3.transpose().dot(dZ3) * derivReLu(z2)   #[m, 16] [10, m]
    dW2 = a1.dot(dZ2.transpose())   #dW2[16, 10]
'''
def back_prop(z1, w1, a1, z2, w2, a2, w3, a3, x, y):
    one_hot_y = one_hot_encoding(y)
    m = one_hot_y.shape[1]
    dZ3 = a3 - one_hot_y                                #dZ3[10, m]
    dW3 = 1 / m * dZ3.dot(a2.T)                         #dW3[10, 16]
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)    #db3[10, 1]
    dZ2 = w3.T.dot(dZ3) * derivReLu(z2)                 #[16, 10].[10, m] = [16, m] * [16, m] 
    dW2 = 1 / m * dZ2.dot(a1.T)                         #[16, m].[m, 16] = [16, 16]
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)    #[16, 1]
    dZ1 = w2.T.dot(dZ2) * derivReLu(z1)                 #[16, 16].[16, m] = [16, m] * [16, m]
    dW1 = 1 / m * dZ1.dot(x.T)                          #[16, m].[m, 784] = [16, 784]
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)    #[16, 1]
    return dW1, db1, dW2, db2, dW3, db3

def updateParams(w1, b1, w2, b2, w3, b3, dW1, db1, dW2, db2, dW3, db3, step):
    w1 = w1 - step * dW1
    b1 = b1 - step * db1
    w2 = w2 - step * dW2
    b2 = b2 - step * db2
    w3 = w3 - step * dW3
    b3 = b3 - step * db3

    return w1, b1, w2, b2, w3, b3

def getPrediction(a3):
    return np.argmax(a3, 0)     #[1, m]

def getAccuracy(prediction, y):
    print(prediction, y)
    correct = prediction == y
    accuracy = 1 / prediction.size * np.sum(correct)
    return accuracy

def gradientDescent(x, y, iteration, input_features, step):
    w1, b1, w2, b2, w3, b3 = initParams(input_features, 5)

    for i in range(iteration):
        z1, a1, z2, a2, z3, a3 = forw_prop(w1, b1, w2, b2, w3, b3, x)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(z1, w1, a1, z2, w2, a2, w3, a3, x, y)
        w1, b1, w2, b2, w3, b3 = updateParams(w1, b1, w2, b2, w3, b3, dW1, db1, dW2, db2, dW3, db3, step)
        if(i % 10 == 0):
            prediction = getPrediction(a3)
            print(f'Iteration: {i}')
            accuracy = getAccuracy(prediction, y)
            print(f'Accuracy: {accuracy}')

    return w1, b1, w2, b2, w3, b3

