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
            raise ValueError('Please use p OR control OR mean,std')

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

    def run_test(control: np.array,
                 test: np.array) -> float:
        """
        Calculate p-value for independent groups

        :param control: Control group data
        :param test: Test group date
        :return: p-value
        """

        mean_control = np.mean(control)
        mean_test = np.mean(test)
        std_control = np.std(control, ddof=1)
        std_test = np.std(test, ddof=1)

        n_control = len(control)
        n_test = len(test)

        t_stat = (mean_test - mean_control) / np.sqrt((std_control ** 2 / n_control) + (std_test ** 2 / n_test))

        df = n_control + n_test - 2

        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))

        return p_value

    def calculate_power(control: np.array,
                        test: np.array,
                        alpha: float = 0.05,
                        is_one_side: float = 1
                        ) -> float:
        """
        Power calculation

        :param control: Control data
        :param test: Test data
        :param alpha: Significance level
        :return: Power
        """

        mean_control = np.mean(control)
        mean_test = np.mean(test)

        effect_size = mean_test - mean_control

        std_control = np.std(control, ddof=1)
        std_test = np.std(test, ddof=1)
        pooled_std = np.sqrt((std_control ** 2 + std_test ** 2) / 2)

        n_control = len(control)
        n_test = len(test)

        n = (n_control + n_test) / 2

        if is_one_side == 1:
            z_alpha = norm.ppf(1 - alpha)
        else:
            z_alpha = norm.ppf(1 - alpha / 2)

        z_beta = (effect_size / pooled_std) * np.sqrt(n) - z_alpha

        power = norm.cdf(z_beta)

        return power