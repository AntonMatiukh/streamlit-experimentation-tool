"""File providing functions for stat tests"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm, ttest_ind
from scipy import stats
# from numba import jit
from numba import njit, prange
from src.parallel.runner import ParallelRunner


# TODO: Think about sequential approach
def sample_ratio_mismatch(observed_counts: list[int],
                          expected_counts: list[int]) -> float:
    """
    Function to check for A/B/n sample ratio mismatch using chi-square test

    :param observed_counts: A list of observed counts for each group.
    :param expected_counts: A list of expected counts for each group.
    :return: True if the chi-square test indicates a significant mismatch, False otherwise
    """

    observed_counts = np.array(observed_counts)
    expected_counts = np.array(expected_counts)

    _, p_value, _, _ = chi2_contingency([observed_counts, expected_counts])

    return p_value


def calc_min_sample_size_ttest_manual(p: float,
                                      mean: float,
                                      std: float,
                                      uplift: float,
                                      alpha: float = 0.05,
                                      power: float = 0.8,
                                      is_one_side: float = 1,
                                      r: float = 0.5) -> int:
    """
    Function to calculate minimum sample size for ztest

    :param p: Probability for binomial data
    :param mean: Mean for continious metrics
    :param std: STD for continius metrics
    :param uplift: Uplift
    :param alpha: alpha
    :param power: power
    :param is_one_side: 1 or 2 side
    :param r: Proportion of
    :return: Minimum sample size
    """

    if is_one_side == 1:
        m = (norm.ppf(q=1 - alpha) + norm.ppf(q=power)) ** 2
    else:
        m = (norm.ppf(q=1 - alpha / 2) + norm.ppf(q=power)) ** 2

    if p is not None:
        es = p * (uplift - 1)
        var_1 = p * (1 - p)
        var_2 = p * uplift * (1 - p * uplift)
        n = ((m / (es ** 2)) * (var_1 / r + var_2 / (1 - r))) / 2
    else:
        sd = std
        es = uplift * mean - mean
        n = (m * ((sd ** 2) / (r * (1 - r))) / (es ** 2)) / 2

    return n

def bootstrap_parallel(control: pd.Series,
                       test: pd.Series,
                       n_bootstraps=1_000,
                       alternative='greater') -> float:
    """

    :param control: Control group data
    :param test: Test group data
    :param n_bootstraps: Count of bootstrap iteration
    :param alternative: Greater/less or two-sided test
    :return: p-value of bootstrap test
    """

    p_value = 0

    for _ in range(n_bootstraps):
        control_b = np.random.choice(control, len(control))
        test_b = np.random.choice(test, len(test))
        p_value += 1 if np.mean(control_b) < np.mean(test_b) else 0

    if alternative == 'two-sided':
        p_value = 2 * min(p_value, n_bootstraps - p_value) / n_bootstraps
    elif alternative == 'greater':
        p_value = p_value / n_bootstraps
    elif alternative == 'less':
        p_value = 1 - p_value / n_bootstraps

    return p_value

def bootstrap(control: pd.Series,
              test: pd.Series,
              n_bootstraps=1_000,
              alternative='greater') -> float:
    """

    :param control: Control group data
    :param test: Test group data
    :param n_bootstraps: Count of bootstrap iteration
    :param alternative: Greater/less or two-sided test
    :return: p-value of bootstrap test
    """
    p_value = 0

    for _ in range(n_bootstraps):
        control_b = np.random.choice(control, len(control))
        test_b = np.random.choice(test, len(test))
        p_value += 1 if np.mean(control_b) < np.mean(test_b) else 0

    if alternative == 'two-sided':
        p_value = 2 * min(p_value, n_bootstraps - p_value) / n_bootstraps
    elif alternative == 'greater':
        p_value = p_value / n_bootstraps
    elif alternative == 'less':
        p_value = 1 - p_value / n_bootstraps

    return p_value


@njit
def bootstrap_njit(control: pd.Series,
              test: pd.Series,
              n_bootstraps=1_000,
              alternative='greater') -> float:
    """

    :param control: Control group data
    :param test: Test group data
    :param n_bootstraps: Count of bootstrap iteration
    :param alternative: Greater/less or two-sided test
    :return: p-value of bootstrap test
    """
    p_value = 0

    for _ in prange(n_bootstraps):
        control_b = np.random.choice(control, len(control))
        test_b = np.random.choice(test, len(test))
        p_value += 1 if np.mean(control_b) < np.mean(test_b) else 0

    if alternative == 'two-sided':
        p_value = 2 * min(p_value, n_bootstraps - p_value) / n_bootstraps
    elif alternative == 'greater':
        p_value = p_value / n_bootstraps
    elif alternative == 'less':
        p_value = 1 - p_value / n_bootstraps

    return p_value


def bucketization(control: pd.Series,
                  test: pd.Series,
                  buckets_count=100,
                  alternative='greater') -> float:
    """

    :param control: Control group data
    :param test: Test group data
    :param buckets_count: Count of buckets
    :param alternative: Greater/less or two-sided test
    :return: p-value of bucketization test
    """

    if len(control) < buckets_count or len(test) < buckets_count:
        raise ValueError("Data amount issue, please use larger df")

    k_control = len(control) // buckets_count
    k_test = len(test) // buckets_count

    control_shuffled = control.sample(frac=1).reset_index(drop=True)
    test_shuffled = test.sample(frac=1).reset_index(drop=True)

    buckets_control = []
    buckets_test = []

    for i in range(buckets_count):
        buckets_control.append(np.mean(control_shuffled[i * k_control: (i + 1) * k_control]))
        buckets_test.append(np.mean(test_shuffled[i * k_test: (i + 1) * k_test]))

    return ttest_ind(buckets_test, buckets_control, alternative=alternative).pvalue


# CTR tests

def linearization(clicks_control: pd.Series,
                  views_control: pd.Series,
                  clicks_test: pd.Series,
                  views_test: pd.Series) -> tuple:
    """

    :param clicks_control: Control group clicks
    :param views_control: Control group views
    :param clicks_test: Test group clicks
    :param views_test: Test group views
    :return: Linearized vectors
    """

    ctr = sum(clicks_control) / sum(views_control)
    linearized_control = clicks_control - ctr * views_control
    linearized_test = clicks_test - ctr * views_test

    return linearized_control, linearized_test


def delta_method(clicks_control: pd.Series,
                 views_control: pd.Series,
                 clicks_test: pd.Series,
                 views_test: pd.Series,
                 alternative='greater') -> float:
    """

    :param clicks_control: Control group clicks
    :param views_control: Control group views
    :param clicks_test: Test group clicks
    :param views_test: Test group views
    :param alternative: Greater/less or two-sided test
    :return: Delta method p-value
    """

    n_control, n_test = len(clicks_control), len(clicks_test)

    clicks_control_mean, clicks_test_mean = np.mean(clicks_control), np.mean(clicks_test)
    clicks_control_var, clicks_test_var = np.var(clicks_control), np.var(clicks_test)
    views_control_mean, views_test_mean = np.mean(views_control), np.mean(views_test)
    views_control_var, views_test_var = np.var(views_control), np.var(views_test)

    control_cov = np.cov(clicks_control, views_control)
    test_cov = np.cov(clicks_test, views_test)

    control_var = clicks_control_var / views_control_mean ** 2 \
                  + views_control_var * clicks_control_mean ** 2 / views_control_mean ** 4 \
                  - 2 * clicks_control_mean / views_control_mean ** 3 * control_cov
    test_var = clicks_test_var / views_test_mean ** 2 \
               + views_test_var * clicks_test_mean ** 2 / views_test_mean ** 4 \
               - 2 * clicks_test_mean / views_test_mean ** 3 * test_cov

    ctr_control = sum(clicks_control) / sum(views_control)
    ctr_test = sum(clicks_test) / sum(views_test)

    z = (ctr_test - ctr_control) / np.sqrt(control_var / n_control + test_var / n_test)

    if alternative == 'two-sided':
        p_value = 2 * min(norm(0, 1).cdf(z), 1 - norm(0, 1).cdf(z))
    elif alternative == 'greater':
        p_value = norm(0, 1).cdf(z)
    elif alternative == 'less':
        p_value = 1 - norm(0, 1).cdf(z)

    return p_value


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


def permutation_test(group1, group2, n_permutations=2_000):
    """
    Perform a permutation test to compare the means of two groups.

    Parameters:
    - group1: array-like, first group of observations
    - group2: array-like, second group of observations
    - n_permutations: int, number of permutations to perform

    Returns:
    - p_value: float, p-value of the test
    """
    observed_diff = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    count = 0

    for _ in range(n_permutations):
        np.random.shuffle(combined)
        permuted_group1 = combined[:len(group1)]
        permuted_group2 = combined[len(group1):]
        permuted_diff = np.mean(permuted_group1) - np.mean(permuted_group2)
        if np.abs(permuted_diff) >= np.abs(observed_diff):
            count += 1

    p_value = count / n_permutations
    return p_value
