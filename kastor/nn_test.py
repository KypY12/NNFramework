from kastor.globals_ import *
from kastor.nn_algorithms import feedforward
from kastor.utils import one_hot, is_equal


def test_network(instances, actual_values, w_matrices, b_matrices, act_matrices, use_one_hot=False):
    count_valid = 0
    for instance, actual_value in tqdm(zip(instances, actual_values)):
        x = np.array([instance]).transpose()
        feedforward(w_matrices, b_matrices, x, act_matrices)

        output = act_matrices[-1][0]
        if use_one_hot:
            output = one_hot(output)

        t = actual_value.transpose()[0]

        if is_equal(output, t):
            count_valid += 1

    return (count_valid * 100) / len(instances)