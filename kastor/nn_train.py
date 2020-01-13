from kastor.creators import *
from kastor.globals_ import *
from kastor.nn_algorithms import feedforward, back_propagation
from kastor.nn_updates import update_w_b, add_gradients_batch, prepare_gradients


def train_network(instances, actual_values,
                  lr, batch_size,
                  w_matrices, b_matrices, act_matrices,
                  momentum_friction=-1, nesterov=False,
                  l1_lambda=0.0,
                  l2_lambda=0.0,
                  RMSprop_parameter=-1, adadelta=False,
                  adam_beta1=-1, adam_beta2=-1,
                  adagrad=False):
    count = 0
    m_matrices = []

    RMSprop_w = []
    RMSprop_b = []

    adam_w_m = []
    adam_w_v = []
    adam_b_m = []
    adam_b_v = []

    adagrad_w_sums = []
    adagrad_b_sums = []

    if momentum_friction != -1:
        m_matrices = create_w_like_matrices(w_matrices)

    elif RMSprop_parameter != -1:
        RMSprop_w = create_w_like_matrices(w_matrices)
        RMSprop_b = create_b_like_matrices(b_matrices)
        # for matrice in w_matrices:
        #     previous_gradients.append(np.zeros(matrice.shape))

    elif adam_beta1 != -1 and adam_beta2 != -1:
        adam_w_m = create_w_like_matrices(w_matrices)
        adam_w_v = create_w_like_matrices(w_matrices)

        adam_b_m = create_b_like_matrices(b_matrices)
        adam_b_v = create_b_like_matrices(b_matrices)

    elif adagrad:
        adagrad_w_sums = create_w_like_matrices(w_matrices)
        adagrad_b_sums = create_b_like_matrices(b_matrices)

    # Iteratiile antrenamentului incep aici
    for instance, actual_value in zip(instances, actual_values):
        x = np.array([instance]).transpose()
        act_matrices = feedforward(w_matrices, b_matrices, x, act_matrices)
        t = actual_value

        # Punem si stratul de input (BKP va avea nevoie de el la calculul gradientilor intre primul si al doilea strat)
        temp = [[x]] + act_matrices

        if batch_size == 0 or batch_size == 1:
            # Online training
            gradients_w, gradients_b = back_propagation(temp, w_matrices, t)

            w_matrices, b_matrices, m_matrices, \
            RMSprop_w, RMSprop_b, \
            adam_w_m, adam_w_v, adam_b_m, adam_b_v, \
            adagrad_w_sums, adagrad_b_sums \
                = update_w_b(w_matrices, b_matrices, gradients_w,
                             gradients_b, lr,
                             count, adam_w_m, adam_w_v, adam_b_m, adam_b_v, adam_beta1, adam_beta2,  # pentru Adam
                             adagrad_w_sums, adagrad_b_sums,
                             m_matrices, momentum_friction, nesterov,  # pentru momentum
                             RMSprop_w, RMSprop_b, RMSprop_parameter,  # pentru RMS Prop
                             adadelta,
                             l1_lambda, l2_lambda, len(instances)  # pentru L1 si L2 regularizare
                             )
        else:
            if count % batch_size == 0:
                gradients_w, gradients_b = back_propagation(temp, w_matrices, t)
            else:
                new_gradients_w, new_gradients_b = back_propagation(temp, w_matrices, t)
                gradients_w, gradients_b = add_gradients_batch(gradients_w, gradients_b, new_gradients_w,
                                                               new_gradients_b)

            if (count + 1) % batch_size == 0:
                gradients_w, gradients_b = prepare_gradients(gradients_w, gradients_b, batch_size)

                w_matrices, b_matrices, m_matrices, \
                RMSprop_w, RMSprop_b, \
                adam_w_m, adam_w_v, adam_b_m, adam_b_v, \
                adagrad_w_sums, adagrad_b_sums \
                    = update_w_b(w_matrices, b_matrices, gradients_w,
                                 gradients_b, lr,
                                 count, adam_w_m, adam_w_v, adam_b_m, adam_b_v, adam_beta1, adam_beta2,  # pentru Adam
                                 adagrad_w_sums, adagrad_b_sums,
                                 m_matrices, momentum_friction, nesterov,  # pentru momentum
                                 RMSprop_w, RMSprop_b, RMSprop_parameter,  # pentru RMS Prop
                                 adadelta,
                                 l1_lambda, l2_lambda, batch_size  # pentru L1 si L2 regularizare
                                 )

        count += 1

    return w_matrices, b_matrices
