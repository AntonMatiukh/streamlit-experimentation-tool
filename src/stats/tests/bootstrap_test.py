"""Class for simulation tests"""
from abc import ABC

import numpy as np
from src.stats.tests.simulation_test import SimulationTest
from src.parallel.runner import ParallelRunner


class BootstrapTest(SimulationTest, ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _perform_bootstrap_test_iteration(control: np.ndarray, test: np.ndarray) -> int:
        """
        1 bootstrapped iteration of the stat test
        """

        control_b = np.random.choice(control, len(control), replace=True)
        test_b = np.random.choice(test, len(test), replace=True)

        return 1 if np.mean(control_b) < np.mean(test_b) else 0

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

        runner = ParallelRunner(n_iterations=self.n_iterations, n_jobs=-1)
        results = runner.run(func=self._perform_bootstrap_test_iteration,
                             control=control,
                             test=test
                             )

        p_value = sum(results)

        if self.alternative == 'two-sided':
            p_value = 2 * min(p_value, len(results) - p_value) / len(results)
        elif self.alternative == 'greater':
            p_value = 1 - p_value / len(results)
        elif self.alternative == 'less':
            p_value = p_value / len(results)

        return p_value

