from kastor.globals_ import *


def create_hidden_layers(size_list, act_func_list, functs_for_error):
    count = len(size_list)
    hidd_layers = []

    for index, size, activation_function, func_for_error in \
            zip(range(0, count), size_list, act_func_list[:-1], functs_for_error[:-1]):
        hidd_layers.append([np.array([[0] * int(size)]).transpose(), activation_function, func_for_error])
    return hidd_layers


def create_weight_matrices(size_list, input_layer_size, output_layer_size):
    count = len(size_list)
    weights_matrices = []
    prev_size = input_layer_size
    for index, size in zip(range(0, count), size_list):
        weights_matrices.append(np.array([[0] * prev_size] * int(size)))
        prev_size = int(size)

    # Intre ultimul hidden layer si layerul de output
    weights_matrices.append(np.array([[0] * prev_size] * output_layer_size))
    return weights_matrices


def create_bias_matrices(size_list, output_layer_size):
    count = len(size_list)

    biases_matrices = []

    for index, size in zip(range(0, count), size_list):
        biases_matrices.append(np.array([[0] * int(size)]).transpose())

    # Pentru layerul de output
    biases_matrices.append(np.array([[0] * output_layer_size]).transpose())
    return biases_matrices


def create_momentum_matrices(w_matrices):
    return [np.zeros(w_matrices[matrix_index].shape) for matrix_index in range(0, len(w_matrices))]