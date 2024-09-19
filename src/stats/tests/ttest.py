"Class for ttest"

from src.stats.tests.abstract_test import StatTest
from scipy.stats import norm
import numpy as np
from scipy import stats
import pandas as pd


class TTest(StatTest):
    def __init__(self, power=0.8, alpha=0.05, alternative='greater'):
        self.power = power
        self.alpha = alpha
        self.alternative = alternative

    def run_test(self,
                 control: np.array,
                 test: np.array,
                 is_one_side: float = 1
                 ) -> float:
        """
        Calculate p-value for independent groups

        :param is_one_side:
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

        p_value = (1 - stats.t.cdf(np.abs(t_stat), df=df))

        if is_one_side == 0:
            p_value *= 2

        return p_value

    def min_sample_size(self,
                        uplift: float,
                        control: np.ndarray = None,
                        p: float = None,
                        mean: float = None,
                        std: float = None,
                        is_one_side: float = 1,
                        r: float = 0.5) -> int:
        """
        Function to calculate minimum sample size for ztest

        :param control: Raw data
        :param p: Probability for binomial data
        :param mean: Mean for continious metrics
        :param std: STD for continius metrics
        :param uplift: Uplift
        :param is_one_side: 1 or 2 side
        :param r: Proportion of
        :return: Minimum sample size
        """

        if p is None and (mean is None or std is None) and control is None:
            raise ValueError('Please add necessary parameters')
        elif ((p is not None and control is not None)
              or (p is not None and (mean is not None or std is not None))
              or (control is not None and (mean is not None or std is not None))):
            raise ValueError('Please use p OR control OR mean,std')

        if control is not None:
            mean, std = np.mean(control), np.std(control)

        if is_one_side == 1:
            m = (norm.ppf(q=1 - self.alpha) + norm.ppf(q=self.power)) ** 2
        else:
            m = (norm.ppf(q=1 - self.alpha / 2) + norm.ppf(q=self.power)) ** 2

        if p is not None:
            es = p * (uplift - 1)
            var_1 = p * (1 - p)
            var_2 = p * uplift * (1 - p * uplift)
            n = ((m / (es ** 2)) * (var_1 / r + var_2 / (1 - r))) / 2
        else:
            sd = std
            es = uplift * mean - mean
            n = (m * ((sd ** 2) / (r * (1 - r))) / (es ** 2)) / 2

        return round(n)

    def min_sample_size_df(self,
                           control: np.ndarray = None,
                           p: float = None,
                           mean: float = None,
                           std: float = None,
                           is_one_side: float = 1,
                           r: float = 0.5) -> pd.DataFrame:
        test_results = []
        for i in range(1, 21, 2):
            test_results.append({'test_name': 'ttest',
                                 'uplift': i,
                                 'n': self.min_sample_size(uplift=1 + i / 100,
                                                           control=control,
                                                           p=p,
                                                           mean=mean,
                                                           std=std,
                                                           is_one_side=is_one_side,
                                                           r=r)})

        return pd.DataFrame(test_results)

    def calculate_power(self,
                        control: np.array,
                        test: np.array,
                        is_one_side: float = 1
                        ) -> float:
        """
        Power calculation

        :param is_one_side:
        :param control: Control data
        :param test: Test data
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
            z_alpha = norm.ppf(1 - self.alpha)
        else:
            z_alpha = norm.ppf(1 - self.alpha / 2)

        z_beta = (effect_size / pooled_std) * np.sqrt(n) - z_alpha

        power = norm.cdf(z_beta)

        return power
