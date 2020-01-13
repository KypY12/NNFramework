from kastor.globals_ import *


def one_hot(output):
    tmp = output.transpose()[0]
    max_z = output.max()
    result = []
    for i in tmp:
        if i == max_z:
            result.append(1)
        else:
            result.append(0)

    return result


def is_equal(a, b):
    # print(a)
    # print(b)
    for index in range(0, len(a)):
        if abs(a[index] - b[index]) > 1:
            return False
    return True


# def is_equal(a, b):
#     for index in range(0, len(a)):
#         if a[index] != b[index]:
#             return False
#     return True


