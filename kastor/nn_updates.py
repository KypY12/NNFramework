def get_current_momentum(lr, gradients_w, prev_momentums, friction):
    current_momentum = []
    # print(len(prev_momentums))
    for matrix_index in range(0, len(prev_momentums)):
        # print(prev_momentums[matrix_index].shape)
        # print(gradients_w[matrix_index].shape)
        # print(prev_momentums[matrix_index])
        current_momentum.append(friction * prev_momentums[matrix_index] - lr * gradients_w[matrix_index])
    return current_momentum


def update_w_b(w_matrices, b_matrices, gradients_w, gradients_b, lr, prev_momentums=[], friction=-1, l2_lambda=0.0, l2_n=0):
    current_momentum = []
    is_momentum = False
    if len(prev_momentums) > 0 and friction != -1:
        is_momentum = True
        current_momentum = get_current_momentum(lr, gradients_w, prev_momentums, friction)

    for matrix_index in range(0, len(w_matrices)):
        # Prima parte
        if l2_lambda != 0 and l2_n != 0:
            w_matrices[matrix_index] *= (1 - lr * (l2_lambda / l2_n))

        # A doua parte
        if is_momentum:
            w_matrices[matrix_index] += current_momentum[matrix_index]
        else:
            w_matrices[matrix_index] -= lr * gradients_w[matrix_index]

    for matrix_index in range(0, len(b_matrices)):
        b_matrices[matrix_index] -= (lr * gradients_b[matrix_index])

    return current_momentum


def add_gradients_batch(old_gradients_w, old_gradients_b, new_gradients_w, new_gradients_b):
    for index in range(0, len(old_gradients_w)):
        old_gradients_w[index] = old_gradients_w[index] + new_gradients_w[index]

    for index in range(0, len(old_gradients_b)):
        old_gradients_b[index] = old_gradients_b[index] + new_gradients_b[index]


def prepare_gradients(gradients_w, gradients_b, batch_len):
    for index in range(0, len(gradients_w)):
        gradients_w[index] = gradients_w[index] / batch_len

    for index in range(0, len(gradients_b)):
        gradients_b[index] = gradients_b[index] / batch_len
