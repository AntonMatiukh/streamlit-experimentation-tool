import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu,norm


def sample_ration_mismatch(observed_counts: list[int], expected_counts: list[int]) -> bool:
    """
    Function to check for A/B/n sample ratio mismatch using chi-square test

    :param observed_counts: A list of observed counts for each group.
    :param expected_counts: A list of expected counts for each group.
    :return: True if the chi-square test indicates a significant mismatch, False otherwise
    """

    observed_counts = np.array(observed_counts)
    expected_counts = np.array(expected_counts)

    _, p_value, _, _ = chi2_contingency([observed_counts, expected_counts])

    return p_value < 0.05


#TODO: update for A/B/n test
def calc_min_sample_size_ttest_manual(p: float, mean: float, std: float, uplift: float,
                                      alpha: float=0.05, power: float=0.8, is_one_side:float=1, r:float=0.5):
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
        M = (norm.ppf(q=1 - alpha) + norm.ppf(q=power)) ** 2
    else:
        M = (norm.ppf(q=1 - alpha / 2) + norm.ppf(q=power)) ** 2

    if p is not None:
        es = p*(uplift - 1)
        var_1 = p*(1-p)
        var_2 = p*uplift*(1 - p*uplift)
        n = ((M / (es ** 2)) * (var_1/r + var_2/(1-r))) / 2
    else:
        sd = std
        es = uplift * mean - mean
        n = (M * ((sd ** 2) / (r*(1-r))) / (es ** 2)) / 2

    return n




