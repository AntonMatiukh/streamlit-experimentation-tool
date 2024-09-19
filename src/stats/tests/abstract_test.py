"Abstract class for statistical tests"

from abc import ABC, abstractmethod
import numpy as np


class StatTest(ABC):
    @abstractmethod
    def run_test(self, control: np.ndarray, test: np.ndarray) -> float:
        """
        Abstract method to calculate p-value
        """
        pass

    @abstractmethod
    def min_sample_size(self,
                        control: np.ndarray,
                        uplift: float,
                        r: float) -> int:
        """
        Abstract method to calculate min sample size for statistical test
        """
        pass

    @abstractmethod
    def calculate_power(self,
                        control: np.ndarray,
                        test: np.ndarray,
                        alpha: float) -> float:
        """
        Abstract method to calculate power for statistical test
        """
        pass
