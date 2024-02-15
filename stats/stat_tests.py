import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu


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



