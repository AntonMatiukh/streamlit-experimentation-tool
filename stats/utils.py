"""Helpers & utils for testing"""

import numpy as np
import pandas as pd
# import logging as lg
from scipy import stats


# Function for CUPED method to reduce variance using historical data
def cuped(control_before: pd.Series,
          control_after: pd.Series,
          test_before: pd.Series,
          test_after: pd.Series) -> tuple:
    """

    :param control_before: Control group vector before test
    :param control_after: Control group vector after test
    :param test_before: Test group vector before test
    :param test_after: Test group vector after test
    :return: Control and test CUPED vectors
    """
    cuped_control = control_after - (control_before - np.mean(control_before)) \
                    * (np.cov(control_after, control_before)[0][1] / np.var(control_before))
    cuped_test = test_after - (test_before - np.mean(test_before)) \
                 * (np.cov(test_after, test_before)[0][1] / np.var(test_before))

    print('Control variance changed on %: '+ \
           str(round(np.var(cuped_control) * 100 / np.var(control_after) - 100,1)))
    print('Test variance changed on %: ' + \
           str(round(np.var(cuped_test) * 100 / np.var(test_after) - 100,1)))

    return cuped_control, cuped_test


# Function to cut/replace outliers using quantile or IQR approaches
def outliers_cut_using_quantile(control: pd.Series,
                                test: pd.Series,
                                historical: pd.Series,
                                q: float,
                                cut_or_replace: str,
                                method: str) -> tuple:
    """

    :param control: Control group data
    :param test: Test group data
    :param historical: Historical data
    :param q: Quantile to replace
    :param cut_or_replace: Cut outlier data or replace it with quantile value
    :param method: Use quatile approach or IQR
    :return: Control and Test vectors without outliers
    """
    if method == 'quantile':
        q = np.quantile(historical, q)
        if cut_or_replace == 'replace':
            control_wo, test_wo = np.where(control > q, q, control), np.where(test > q, q, test)
        elif cut_or_replace == 'cut':
            control_wo, test_wo = control[control <= q], test[test <= q]
    elif method == 'iqr':
        q1 = np.quantile(historical, 0.25)
        q3 = np.quantile(historical, 0.75)
        iqr = q3 - q1

        if cut_or_replace == 'replace':
            control_wo = np.where(control > q3 + 1.5*iqr, q3 + 1.5*iqr, control)
            control_wo = np.where(control_wo < q1 - 1.5*iqr, q1 - 1.5*iqr, control_wo)
            test_wo = np.where(test > q3 + 1.5 * iqr, q3 + 1.5 * iqr, test)
            test_wo = np.where(test_wo < q1 - 1.5 * iqr, q1 - 1.5 * iqr, test_wo)
        elif cut_or_replace == 'cut':
            control_wo = control[(control > (q1 - 1.5*iqr)) & (control < (q3 + 1.5*iqr))]
            test_wo = test[(test > (q1 - 1.5*iqr)) & (test < (q3 + 1.5*iqr))]


    print('Control variance changed on %: ' + \
           str(round(np.var(control_wo) * 100 / np.var(control) - 100,1)))
    print('Test variance changed on %: ' + \
           str(round(np.var(test_wo) * 100 / np.var(test) - 100,1)))

    return control_wo, test_wo


# Transform data
def transform_data(control: pd.Series,
                   test: pd.Series,
                   method: str) -> tuple:
    """

    :param control: Control group data
    :param test: Test group data
    :param method: Method parameter log or box-cox
    :return: Transformed control and test data
    """
    if method == 'log':
        control_t, test_t = np.log(control), np.log(test)
    elif method == 'boxcox':
        control_t, test_t = stats.boxcox(control), stats.boxcox(test)

    return control_t, test_t


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
