import numpy as np

def none(lr):
    return lr

def time_based_decay(lr, iterations, decay = 1):
    lr *= (1 / (1 + decay * iterations))
    return lr

def division_lr(lr, div = 1):
    lr /= div
    return lr

def mul_lr(lr,mul = 1):
    lr = lr * mul
    return lr

def exponential_lr(lr, initial_lr,decay = 1):
    lr = initial_lr * (np.e ** (-decay))
    return lr

def step_based(lr, initial_lr, iterations, droprate, decay = 1):
    def floor(x):
        if x < 1: return 0
        return x
    lr = initial_lr*(decay ** floor((1+iterations)/droprate))
    return lr



