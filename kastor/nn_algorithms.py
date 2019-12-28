from kastor.globals_ import *


def feedforward(w_matrices, b_matrices, x, act_matrices):
    w = w_matrices[0]
    b = b_matrices[0]

    # z = w * x + b
    z = w.dot(x) + b

    # y = act(z)
    activation_function = act_matrices[0][1]
    act_matrices[0][0] = activation_function(z)

    for layer_index in range(1, len(act_matrices)):
        w = w_matrices[layer_index]
        b = b_matrices[layer_index]
        x = act_matrices[layer_index - 1][0]

        z = w.dot(x) + b

        activation_function = act_matrices[layer_index][1]
        act_matrices[layer_index][0] = activation_function(z)

    return act_matrices


def back_propagation(act_matrices, w_matrices, actual_value):
    # Calculez eroarea in ultimul strat (cel de output)
    last_layer = len(act_matrices) - 1
    current_layer = last_layer

    gradients_w = []
    gradients_b = []

    # LOOP (pana prec e input)
    while current_layer > 0:
        # Calculam eroarea stratului precedent --- care e egal cu gradient_b
        # eroarea este o matrice cu 1 coloana si layer.size() linii
        # Daca e ultimul layer calculam eroarea specifica ce este dependenta si de functia de cost
        if current_layer == last_layer:
            error = act_matrices[current_layer][2](act_matrices[current_layer][0], actual_value)
        # Altfel, folosim formula in care apare derivata functiei de activare
        else:
            # weighturile sunt in numar de last_layer-2 deci current_layer+1 devine current_layer pentru ele
            error = error.transpose().dot(w_matrices[current_layer])
            error = np.multiply(error.transpose(), act_matrices[current_layer][2](act_matrices[current_layer][0]))

        gradients_b += [error]
        gradients_w += [error.dot(act_matrices[current_layer - 1][0].transpose())]

        current_layer -= 1

    gradients_w.reverse()
    gradients_b.reverse()
    return gradients_w, gradients_b
