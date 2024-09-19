"""Helpers & utils, currently applicable for functional paradigm"""

from joblib import Parallel, delayed
import numpy as np


def generate_lognormal(mean, std, size):
    """
    Generate a lognormal distribution with the specified mean and std.

    Parameters:
    - mean: float, desired mean of the lognormal distribution
    - std: float, desired standard deviation of the lognormal distribution
    - size: int, number of samples to generate

    Returns:
    - samples: array-like, generated lognormal samples
    """

    mu = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
    sigma = np.sqrt(np.log(1 + (std ** 2 / mean ** 2)))

    samples = np.random.lognormal(mean=mu, sigma=sigma, size=size)
    return samples

def parallel_run(func, n_iterations=2_000, n_jobs=-1, **kwargs):
    results = Parallel(n_jobs=n_jobs)(delayed(func)(**kwargs) for _ in range(n_iterations))
    return results


# Decorator for approximate
def calculate_min_sample_size(power, n_step, alpha, greater_or_less):
    """

    :param power: Power level
    :param n_step: Step amount to iterate for modulations
    :param alpha: Alpha level
    :param greater_or_less: Type of hypothesis
    :return: Minimum sample size
    """
    def decorator(test_func):
        def wrapper(**kwargs):

            n = 0
            current_power = 0

            while current_power < power:

                n += n_step

                current_power = 0

                for _ in range(1_000):

                    if greater_or_less == 'greater':

                        current_power += 1 if test_func(n=n, **kwargs) >= alpha else 0

                    elif greater_or_less == 'less':

                        current_power += 1 if test_func(n=n, **kwargs) <= alpha else 0

                current_power = current_power / 1_000

                print('Sample size: ' + str(n) + ', power level: ' + str(current_power))

            return n

        return wrapper

    return decorator
