from kastor.globals_ import *


def init_weights(weight_init_name, weight_matrices, input_layer_size):
    if weight_init_name:
        if weight_init_name == "normal":
            init_method_normal(weight_matrices, input_layer_size)


def init_bias(bias_init_name, bias_matrices, input_layer_size):
    if bias_init_name:
        if bias_init_name == "normal":
            init_method_normal(bias_matrices, input_layer_size)


# Orice metoda noua se adauga mai jos trebuie adaugata si sus in cele doua functii

def init_method_normal(matrices, input_layer_size):
    prev_size = input_layer_size

    for index in range(0, len(matrices)):
        mean = 0
        std_dev = 1 / np.sqrt(prev_size)
        matrices[index] = np.random.normal(mean, std_dev, matrices[index].shape)

        # Numarul de inputuri intr-un nod din layerul urmator este egal cu
        # numarul de noduri din layerul curent (numarul de linii din matrix)
        prev_size = len(matrices[index])