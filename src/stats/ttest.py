"Class for ttest"

from src.stats.abstract_test import StatTest
from scipy.stats import chi2_contingency, norm, ttest_ind
import numpy as np
from scipy import stats


class TTest(StatTest):
    def min_sample_size(uplift: float,
                        control: np.ndarray = None,
                        p: float = None,
                        mean: float = None,
                        std: float = None,
                        alpha: float = 0.05,
                        power: float = 0.8,
                        is_one_side: float = 1,
                        r: float = 0.5) -> int:
        """
        Function to calculate minimum sample size for ztest

        :param control: Raw data
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

        if (p is None and (mean is None or std is None) and control is None):
            raise ValueError('Please add necessary parameters')
        elif ((p is not None and control is not None)
            or (p is not None and (mean is not None or std is not None))
            or (control is not None and (mean is not None or std is not None))):
            raise ValueError('Please p OR control OR mean,std')

        if control is not None:
            mean, std = np.mean(control), np.std(control)

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



