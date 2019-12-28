from kastor.creators import create_momentum_matrices
from kastor.globals_ import *
from kastor.nn_algorithms import feedforward, back_propagation
from kastor.nn_updates import update_w_b, add_gradients_batch, prepare_gradients


def train_network(instances, actual_values,
                  lr, batch_size,
                  w_matrices, b_matrices, act_matrices,
                  momentum_friction=-1,
                  l2_lambda=0.0):

    count = 0
    m_matrices = []
    if momentum_friction != -1:
        m_matrices = create_momentum_matrices(w_matrices)

    for instance, actual_value in tqdm(zip(instances, actual_values), position=1):
        x = np.array([instance]).transpose()
        act_matrices = feedforward(w_matrices, b_matrices, x, act_matrices)
        t = actual_value

        # Punem si stratul de input (BKP va avea nevoie de el la calculul gradientilor intre primul si al doilea strat)
        temp = [[x]] + act_matrices

        if batch_size == 0 or batch_size == 1:
            # Online training
            gradients_w, gradients_b = back_propagation(temp, w_matrices, t)
            m_matrices = update_w_b(w_matrices, b_matrices, gradients_w, gradients_b, lr,
                                    m_matrices, momentum_friction,  # pentru momentum
                                    l2_lambda, len(instances)       # pentru L2 regularizare
                                    )
        else:
            if count % batch_size == 0:
                gradients_w, gradients_b = back_propagation(temp, w_matrices, t)
            else:
                new_gradients_w, new_gradients_b = back_propagation(temp, w_matrices, t)
                add_gradients_batch(gradients_w, gradients_b, new_gradients_w, new_gradients_b)

            if (count+1) % batch_size == 0:
                prepare_gradients(gradients_w, gradients_b, batch_size)
                m_matrices = update_w_b(w_matrices, b_matrices, gradients_w, gradients_b, lr,
                                        m_matrices, momentum_friction,  # pentru momentum
                                        l2_lambda, batch_size           # pentru L2 regularizare
                                        )
        count += 1

    return w_matrices, b_matrices
