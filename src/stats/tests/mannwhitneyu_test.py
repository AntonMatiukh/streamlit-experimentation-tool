"""Class for simulation tests"""
from abc import ABC

import numpy as np
from src.stats.tests.simulation_test import SimulationTest
from src.stats.tests.ttest import TTest
from scipy.stats import mannwhitneyu


class MannWhitneyTest(SimulationTest, ABC):
    def __init__(self):
        super().__init__()

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

        _, p_value = mannwhitneyu(x=test, y=control, alternative=self.alternative)
        # p_value = TTest().run_test(control=control, test=test)

        return p_value
