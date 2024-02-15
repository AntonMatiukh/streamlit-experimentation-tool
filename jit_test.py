# import numpy as np
# from numba import jit
# import time
#
# def sum_of_squares(arr):
#     total = 0
#     for x in arr:
#         total += x ** 2
#     return total
#
# @jit
# def jit_sum_of_squares(arr):
#     total = 0
#     for x in arr:
#         total += x ** 2
#     return total
#
# arr = np.random.randint(0, 100, size=10_000_000)
#
# time_1 = time.time()
# sum_of_squares(arr)
# print(time.time() - time_1)
#
# time_1 = time.time()
# jit_sum_of_squares(arr)
# print(time.time() - time_1)
