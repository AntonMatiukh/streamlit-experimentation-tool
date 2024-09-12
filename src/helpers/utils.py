"""Helpers & utils, currently applicable for functional paradigm"""

from joblib import Parallel, delayed


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
