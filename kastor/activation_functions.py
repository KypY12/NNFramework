from kastor.globals_ import *


# Sigmoid

def act_func_sigmoid(z):
    return np.array([[1 / (1 + np.exp(-i[0]))] for i in z])


def deriv_sigmoid(y_l):
    return y_l * (1 - y_l)


# Softmax

def act_func_softmax(z):
    return np.exp(z) / np.sum(np.exp(z))