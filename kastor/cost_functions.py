import numpy as np


def cost_cross_entropy(y_last, t):
    return y_last - t


# Gresit (trebuie derivata lui MSE)
def cost_mean_squared_error(y_last, t):
    return (1 / len(y_last)) * np.power((y_last-t), 2)
