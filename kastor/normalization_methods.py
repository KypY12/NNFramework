import numpy as np


def normalize_min_max(data_set):
    for data_index in range(0, len(data_set)):
        data_min = np.min(data_set[data_index][0])
        data_max = np.max(data_set[data_index][0])
        data_diff = data_max - data_min
        for data_elem_index in range(0, len(data_set[data_index][0])):
            data_set[data_index][0][data_elem_index] = \
                (data_set[data_index][0][data_elem_index] - data_min) / data_diff

    return data_set


# def normalize_min_max(data_set):
#     global_min, global_max = 0
#
#     for data_index in range(0, len(data_set)):
#         data_min = np.min(data_set[data_index][0])
#         data_max = np.max(data_set[data_index][0])
#         if data_min < global_min:
#             global_min = data_min
#
#         if data_max > global_max:
#             global_max = data_max
#
#     data_diff = global_max - global_min
#
#     for data_index in range(0, len(data_set)):
#         for data_elem_index in range(0, len(data_set[data_index][0])):
#             data_set[data_index][0][data_elem_index] = \
#                 (data_set[data_index][0][data_elem_index] - global_min) / data_diff
#
#     return data_set


def normalize_z_score(data_set):
    for data_index in range(0, len(data_set)):
        data_mean = np.mean(data_set[data_index][0])
        data_std_dev = np.std(data_set[data_index][0])

        for data_elem_index in range(0, len(data_set[data_index][0])):
            data_set[data_index][0][data_elem_index] = \
                (data_set[data_index][0][data_elem_index] - data_mean) / data_std_dev

    return data_set


def normalize_robust(data_set):
    for data_index in range(0, len(data_set)):
        data_q1 = np.quantile(data_set[data_index][0], 0.25)
        data_q2 = np.quantile(data_set[data_index][0], 0.75)
        data_median = np.median(data_set[data_index][0])

        data_diff = data_q2 - data_q1 + 1

        for data_elem_index in range(0, len(data_set[data_index][0])):
            data_set[data_index][0][data_elem_index] = \
                (data_set[data_index][0][data_elem_index] - data_median) / data_diff

    return data_set
