import numpy as np


def get_current_momentum(lr, gradients_w, prev_momentums, friction):
    current_momentum = []
    for matrix_index in range(0, len(prev_momentums)):
        current_momentum.append(friction * prev_momentums[matrix_index] - lr * gradients_w[matrix_index])
    return current_momentum


def get_current_RMSprop(gradients, previous_RMSprop, RMSprop_parameter):
    current_RMSprop = []
    for matrix_index in range(0, len(previous_RMSprop)):
        current_RMSprop.append(
            RMSprop_parameter * previous_RMSprop[matrix_index] +
            (1 - RMSprop_parameter) * np.multiply(gradients[matrix_index], gradients[matrix_index]))
    return current_RMSprop


def get_current_adam_m(gradients, prev_adam_m, beta1):
    current_adam_m = []
    for matrix_index in range(0, len(prev_adam_m)):
        current_adam_m.append(beta1 * prev_adam_m[matrix_index] + (1 - beta1) * gradients[matrix_index])
    return current_adam_m


def get_current_adam_v(gradients, prev_adam_v, beta2):
    current_adam_v = []
    for matrix_index in range(0, len(prev_adam_v)):
        current_adam_v.append(beta2 * prev_adam_v[matrix_index] + (1 - beta2) * np.multiply(gradients[matrix_index],
                                                                                            gradients[matrix_index]))
    return current_adam_v


def get_current_adagrad_sums(gradients, prev_sums):
    current_sums = []
    for matrix_index in range(0, len(prev_sums)):
        current_sums.append(prev_sums[matrix_index] + np.multiply(gradients[matrix_index], gradients[matrix_index]))
    return current_sums


def update_w_b(w_matrices, b_matrices, gradients_w, gradients_b, lr,
               iteration_count, prev_adam_w_m=[], prev_adam_w_v=[], prev_adam_b_m=[], prev_adam_b_v=[],
               adam_beta1=-1, adam_beta2=-1,
               prev_adagrad_w_sums=[], prev_adagrad_b_sums=[],
               prev_momentums=[], friction=-1, nesterov=False,
               prev_w_RMSprop=[], prev_b_RMSprop=[], RMSprop_parameter=-1,
               adadelta=False,
               l1_lambda=0.0, l2_lambda=0.0, l_n=0):

    current_momentum = []

    w_RMSprop = []
    b_RMSprop = []

    w_adam_m = []
    w_adam_v = []

    b_adam_m = []
    b_adam_v = []

    adagrad_w_sums = []
    adagrad_b_sums = []

    is_momentum = False
    is_RMSprop = False
    is_adam = False
    is_adagrad = False

    if len(prev_adagrad_w_sums) > 0 and len(prev_adagrad_b_sums) > 0:
        is_adagrad = True
        adagrad_w_sums = get_current_adagrad_sums(gradients_w, prev_adagrad_w_sums)
        adagrad_b_sums = get_current_adagrad_sums(gradients_b, prev_adagrad_b_sums)

    elif len(prev_momentums) > 0 and friction != -1:
        is_momentum = True
        current_momentum = get_current_momentum(lr, gradients_w, prev_momentums, friction)

    elif len(prev_w_RMSprop) > 0 and len(prev_b_RMSprop) > 0 and RMSprop_parameter != -1:
        is_RMSprop = True
        w_RMSprop = get_current_RMSprop(gradients_w, prev_w_RMSprop, RMSprop_parameter)
        b_RMSprop = get_current_RMSprop(gradients_b, prev_b_RMSprop, RMSprop_parameter)

    elif len(prev_adam_w_m) > 0 and len(prev_adam_w_v) > 0 and len(prev_adam_b_m) > 0 and len(prev_adam_b_v) > 0 \
            and adam_beta1 != -1 and adam_beta2 != -1:
        is_adam = True
        w_adam_m = get_current_adam_m(gradients_w, prev_adam_w_m, adam_beta1)
        w_adam_v = get_current_adam_v(gradients_w, prev_adam_w_v, adam_beta2)
        b_adam_m = get_current_adam_m(gradients_b, prev_adam_b_m, adam_beta1)
        b_adam_v = get_current_adam_v(gradients_b, prev_adam_b_v, adam_beta2)

    for matrix_index in range(0, len(w_matrices)):
        # Prima parte
        if l1_lambda != 0.0 and l_n != 0:
            w_matrices[matrix_index] -= lr * (l1_lambda / l_n) * np.sign(w_matrices[matrix_index])

        elif l2_lambda != 0.0 and l_n != 0:
            w_matrices[matrix_index] *= (1 - lr * (l2_lambda / l_n))

        # A doua parte
        if is_adagrad:
            # Pentru weights
            w_matrices[matrix_index] -= (lr * gradients_w[matrix_index]) / (np.sqrt(adagrad_w_sums) + 10 ** (-8))

            # Pentru bias
            b_matrices[matrix_index] -= (lr * gradients_b[matrix_index]) / (np.sqrt(adagrad_b_sums) + 10 ** (-8))
            # 10**(-8) e epsilon in formula dar nu merita schimbat. E doar ca sa eviti impartirea la 0

        elif is_adam:
            # Pentru weights
            w_m_inter = w_adam_m[matrix_index] / (1 - adam_beta1 ** (iteration_count))
            w_v_inter = w_adam_v[matrix_index] / (1 - adam_beta2 ** (iteration_count))

            w_matrices[matrix_index] -= (lr * w_m_inter) / (np.sqrt(w_v_inter) + 10 ** (-8))

            # Pentru bias
            b_m_inter = b_adam_m[matrix_index] / (1 - adam_beta1 ** (iteration_count))
            b_v_inter = b_adam_v[matrix_index] / (1 - adam_beta2 ** (iteration_count))

            b_matrices[matrix_index] -= (lr * b_m_inter) / (np.sqrt(b_v_inter) + 10 ** (-8))

            # current_momentum[matrix_index] = current_momentum[matrix_index] / (1 - friction ** iteration_count)
            # current_RMSprop[matrix_index] = current_RMSprop[matrix_index] / (1 - RMSprop_parameter ** iteration_count)
            # w_matrices[matrix_index] -= -lr * current_momentum[matrix_index] / (
            #         current_RMSprop[matrix_index] ** (1 / 2) + 10 ** (-8))
            # 10**(-8) e epsilon in formula dar nu merita schimbat. E doar ca sa eviti impartirea la 0

        elif is_momentum:
            # Pentru weights
            w_matrices[matrix_index] += current_momentum[matrix_index]
            if nesterov:
                w_matrices[matrix_index] += friction * (current_momentum[matrix_index] - prev_momentums[matrix_index])

            # Pentru bias
            b_matrices[matrix_index] -= (lr * gradients_b[matrix_index])

        elif is_RMSprop:
            w_value_for_lr = lr
            b_value_for_lr = lr
            if adadelta:
                w_value_for_lr = prev_w_RMSprop[matrix_index]
                b_value_for_lr = prev_b_RMSprop[matrix_index]

            # Pentru weights
            w_matrices[matrix_index] -= (w_value_for_lr * gradients_w[matrix_index]) / (
                        np.sqrt(w_RMSprop[matrix_index]) + 10 ** (-8))

            # Pentru bias
            b_matrices[matrix_index] -= (b_value_for_lr * gradients_b[matrix_index]) / (
                        np.sqrt(b_RMSprop[matrix_index]) + 10 ** (-8))

            # 10**(-8) e epsilon in formula dar nu merita schimbat. E doar ca sa eviti impartirea la 0

        else:
            # Pentru weights
            w_matrices[matrix_index] -= lr * gradients_w[matrix_index]

            # Pentru bias
            b_matrices[matrix_index] -= (lr * gradients_b[matrix_index])

    # for matrix_index in range(0, len(b_matrices)):
    #     b_matrices[matrix_index] -= (lr * gradients_b[matrix_index])

    return w_matrices, b_matrices, current_momentum, \
           w_RMSprop, b_RMSprop, \
           w_adam_m, w_adam_v, b_adam_m, b_adam_v, \
           adagrad_w_sums, adagrad_b_sums


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
