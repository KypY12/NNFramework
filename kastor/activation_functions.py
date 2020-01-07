from kastor.globals_ import *


# Sigmoid

def act_func_sigmoid(z):
    # exp_val = np.exp(z)
    # aux = np.max(z)
    # exp_val = np.exp(z - aux)
    # return exp_val / (1 + exp_val)
    exp_val = np.clip( z, -500, 500 )
    return 1 / (1 + np.exp(-exp_val))
    # return np.array([[1 / (1 + np.exp(-i[0]))] for i in z])


def deriv_sigmoid(y_prev):
    return y_prev * (1 - y_prev)


# Softmax

def act_func_softmax(z):
    aux = np.max(z)
    exp_val = np.exp(z-aux)
    return exp_val / np.sum(exp_val)

# trebuie derivata la softmax


# Linear

def act_funct_linear(z):
    return z


# Relu


def act_funct_relu(z):
    return np.array([[i[0]] if i[0] > 0 else [0] for i in z])


def deriv_relu(y_prev):
    return np.array([[1] if i > 0 else [0] for i in y_prev])

