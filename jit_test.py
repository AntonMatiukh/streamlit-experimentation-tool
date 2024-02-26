import numpy as np
from numba import jit
import time
from scipy import stats
import scipy
from scipy.stats import ttest_ind
from stats.utils import calculate_min_sample_size, cuped
from stats.stat_tests import bucketization


def bootstrap(x,y,n):
    rez = 0
    for _ in range(n):
        x_b = np.random.choice(x, len(x))
        y_b = np.random.choice(y, len(y))
        rez += 1 if np.mean(y_b) > np.mean(x_b) else 0

    # print(1 - rez / n)

    return 1 - rez / n

def poisson_bootstrap(x,y,n):
    poisson_bootstraps = stats.poisson(1).rvs((n, len(x)))
    x_b = np.matmul(x, poisson_bootstraps.T)
    y_b = np.matmul(y, poisson_bootstraps.T)
    rez = np.sum(y_b - x_b < 0) / n
    # print(rez)
    return rez

@jit
def bootstrap_jit(x, y, n):
    rez = 0
    for _ in range(n):
        x_b = np.random.choice(x, len(x))
        y_b = np.random.choice(y, len(y))
        rez += 1 if np.mean(y_b) > np.mean(x_b) else 0

    # print(1 - rez / n)

    return 1 - rez / n

x = stats.norm.rvs(loc=1., scale=10, size=50_000) #np.random.randint(0, 10, size=50_000)
y = stats.norm.rvs(loc=1.12, scale=10, size=50_000) #np.random.randint(10, 10, size=50_000)

# print('Bootstrap')
# time_1 = time.time()
# bootstrap(x,y,2_000)
# print(time.time() - time_1)
#
# print('Poisson bootstrap')
# time_1 = time.time()
# poisson_bootstrap(x,y,2_000)
# print(time.time() - time_1)
#
# print('Bootstrap with jit')
# time_1 = time.time()
# bootstrap_jit(x,y,2_000)
# print(time.time() - time_1)

# Normal Data


# def a_a_test():
#     for _ in range(1000):
#         p_val_aa = []
#         x = stats.genpareto.rvs(1, size=1000)  # np.random.normal(10,4,1000)
#         y = stats.genpareto.rvs(1, size=1000)  # np.random.normal(10,4,1000)
#         p_val_aa.append(bootstrap_jit(x, y, len(x)))
#     return p_val_aa
#
# time_1 = time.time()
# a_a_test()
# print(time.time() - time_1)


# @calculate_min_sample_size(power=0.8, n_step=5_000, alpha=0.05, greater_or_less='less')
# def ttest(p1, p2, n):
#     x = np.random.choice(a=[1,0], p=[p1,1-p1], size=n)
#     y = np.random.choice(a=[1,0], p=[p2,1-p2], size=n)
#     return ttest_ind(y, x, alternative='greater').pvalue
#
# time_1 = time.time()
# ttest(p1=0.19, p2=0.2)
# print(time.time() - time_1)

# x_1 = [1,2,3,4,5,6,7]
# x_2 = [2,4,6,8,10,12,16]
# y_1 = [1,2,3,4,5,6,7]
# y_2 = [1,4,9,16,25,36,50]
#
#
# cuped(x_1,x_2,y_1,y_2)

time_1 = time.time()
for _ in range(1000):
    p_val_aa = []
    x = stats.genpareto.rvs(1, size=20_038)
    y = stats.genpareto.rvs(1, size=20_147)
    p_val_aa.append(bucketization(x, y))
print(time.time() - time_1)





