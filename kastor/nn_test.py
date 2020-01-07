from kastor.globals_ import *
from kastor.nn_algorithms import feedforward
from kastor.utils import one_hot, is_equal


def test_network(instances, actual_values, w_matrices, b_matrices, act_matrices, use_one_hot=False):
    cost_values = []
    count_valid = 0
    local_act_matrices = act_matrices
    for instance, actual_value in zip(instances, actual_values):
        x = np.array([instance]).transpose()
        local_act_matrices = feedforward(w_matrices, b_matrices, x, local_act_matrices)

        output = local_act_matrices[-1][0]
        if use_one_hot:
            output = one_hot(output)

        t = actual_value

        # print(local_act_matrices[-1])
        # print(actual_value)
        # print(output)

        if is_equal(output, t):
            count_valid += 1
        cost_values.append(np.sqrt(np.power((output-t), 2)))

    return (count_valid * 100) / len(instances), np.mean(np.array(cost_values))