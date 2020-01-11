def get_current_momentum(lr, gradients_w, prev_momentums, friction):
    current_momentum = []
    # print(len(prev_momentums))
    for matrix_index in range(0, len(prev_momentums)):
        # print(prev_momentums[matrix_index].shape)
        # print(gradients_w[matrix_index].shape)
        # print(prev_momentums[matrix_index])
        current_momentum.append(friction * prev_momentums[matrix_index] - lr * gradients_w[matrix_index])
    return current_momentum

def get_current_RMSprop(gradients_w, previous_gradients, RMSprop_parameter):
    current_RMSprop = []
    for matrix_index in range(0, len(previous_gradients)):
        current_RMSprop.append(RMSprop_parameter * previous_gradients[matrix_index] + (1 - RMSprop_parameter)*(gradients_w[matrix_index]*gradients_w[matrix_index]))
    return current_RMSprop

def update_w_b(w_matrices, b_matrices, gradients_w, gradients_b, lr, iteration_count, prev_momentums=[], friction=-1, previous_gradients = [], RMSprop_parameter=-1, l2_lambda=0.0, l2_n=0):
    current_momentum = []
    is_momentum = False
    is_RMSprop = False    
    is_Adam = False

    if len(prev_momentums) > 0 and friction != -1:
        is_momentum = True
        current_momentum = get_current_momentum(lr, gradients_w, prev_momentums, friction)

    if len(previous_gradients) > 0 and RMSprop_parameter != -1:
        is_RMSprop = True
        current_RMSprop = get_current_RMSprop(gradients_w, previous_gradients, RMSprop_parameter)

    if is_momentum == True and is_RMSprop == True:
        is_Adam = True

    for matrix_index in range(0, len(w_matrices)):
        # Prima parte
        if l2_lambda != 0 and l2_n != 0:
            w_matrices[matrix_index] *= (1 - lr * (l2_lambda / l2_n))

        # A doua parte
        if is_Adam:
            current_momentum[matrix_index] = current_momentum[matrix_index] / (1 - friction**iteration_count)
            current_RMSprop[matrix_index] = current_RMSprop[matrix_index] / (1 - RMSprop_parameter**iteration_count)        
            w_matrices[matrix_index] -= -lr * current_momentum[matrix_index] / (current_RMSprop[matrix_index]**(1/2) + 10**(-8)) # 10**(-8) e epsilon in formula dar nu merita schimbat. E doar ca sa eviti impartirea la 0
        elif is_momentum:
            w_matrices[matrix_index] += current_momentum[matrix_index]
        elif is_RMSprop:
            w_matrices[matrix_index] -= -lr * gradients_w[matrix_index] / (current_RMSprop[matrix_index]**(1/2) + 10**(-8)) # 10**(-8) e epsilon in formula dar nu merita schimbat. E doar ca sa eviti impartirea la 0
        else:
            w_matrices[matrix_index] -= lr * gradients_w[matrix_index]

    for matrix_index in range(0, len(b_matrices)):
        b_matrices[matrix_index] -= (lr * gradients_b[matrix_index])

    return w_matrices, b_matrices, current_momentum


def add_gradients_batch(old_gradients_w, old_gradients_b, new_gradients_w, new_gradients_b):
    for index in range(0, len(old_gradients_w)):
        old_gradients_w[index] = old_gradients_w[index] + new_gradients_w[index]

    for index in range(0, len(old_gradients_b)):
        old_gradients_b[index] = old_gradients_b[index] + new_gradients_b[index]

    return old_gradients_w, old_gradients_b


def prepare_gradients(gradients_w, gradients_b, batch_len):
    for index in range(0, len(gradients_w)):
        gradients_w[index] = gradients_w[index] / batch_len

    for index in range(0, len(gradients_b)):
        gradients_b[index] = gradients_b[index] / batch_len

    return gradients_w, gradients_b
