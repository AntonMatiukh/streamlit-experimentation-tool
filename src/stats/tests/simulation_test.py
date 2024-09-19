"""Class for simulation tests"""

from abc import abstractmethod, ABC
import numpy as np
from src.stats.tests.abstract_test import StatTest
from src.parallel.runner import ParallelRunner
from src.stats.tests.ttest import TTest


# TODO: Add two-side option
# TODO: Change greater_or_less for alternative
class SimulationTest(StatTest, ABC):
    def __init__(self, n_iterations=2_000, power=0.8, alpha=0.05, alternative='greater'):
        self.n_iterations = n_iterations
        self.power = power
        self.alpha = alpha
        self.alternative = alternative

    def _perform_test_iteration(self, control: np.ndarray, test: np.ndarray) -> int:
        """
        1 bootstrapped iteration of the stat test
        """

        control_b = np.random.choice(control, len(control), replace=True)
        test_b = np.random.choice(test, len(test), replace=True)

        if self.alternative == 'greater':
            return 1 if self.run_test(control=control_b, test=test_b) < self.alpha else 0
        elif self.alternative == 'less':
            return 1 if self.run_test(control=test_b, test=control_b) <= self.alpha else 0

    def calculate_power(self,
                        control: np.ndarray,
                        test: np.ndarray,
                        runner: ParallelRunner = None) -> float:
        """
        Power simulation function
        """

        if runner is None:
            runner = ParallelRunner(n_iterations=self.n_iterations, n_jobs=-1)

        results = runner.run(func=self._perform_test_iteration,
                             control=control,
                             test=test
                             )

        return sum(results) / len(results)

    def min_sample_size(self,
                        control: np.ndarray,
                        uplift: float,
                        is_one_side: int = 1,
                        r: float = 0.5) -> int:
        """
        Sample size simulation function
        """

        n_approximate = TTest(alpha=self.alpha,
                              power=self.power,
                              alternative=self.alternative).min_sample_size(uplift=uplift,
                                                                            control=control,
                                                                            is_one_side=is_one_side,
                                                                            r=r)
        print(2*n_approximate)

        n = 0
        n_approximate = round(1.3 * 2 * n_approximate - n)
        n_step = round(n_approximate // 6)
        power = 0

        for i in range(n, n_approximate + n_step, n_step):
            n += n_step

            control = np.random.choice(control, round(n * r), replace=True)
            test = np.random.choice(control, round(n * (1 - r)), replace=True)
            test *= (1 + np.random.normal(uplift - 1, 0.5, round(n * (1 - r))))

            power = self.calculate_power(control=control, test=test)

            print(i, power)

            if power > self.power:
                return n

        print(f'Current power level is {power}')

        return n

