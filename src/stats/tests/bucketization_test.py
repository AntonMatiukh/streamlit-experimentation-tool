"""Class for simulation tests"""
from abc import ABC

import numpy as np
from src.stats.tests.simulation_test import SimulationTest
from src.stats.tests.ttest import TTest


class BucketizationTest(SimulationTest, ABC):
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

        buckets_count = 200

        if len(control) < buckets_count or len(test) < buckets_count:
            raise ValueError("Data amount issue, please use larger df")

        k_control = len(control) // buckets_count
        k_test = len(test) // buckets_count

        np.random.shuffle(control)
        np.random.shuffle(test)

        buckets_control = []
        buckets_test = []

        for i in range(buckets_count):
            buckets_control.append(np.mean(control[i * k_control: (i + 1) * k_control]))
            buckets_test.append(np.mean(test[i * k_test: (i + 1) * k_test]))

        return TTest(alpha=self.alpha,
                     power=self.power,
                     alternative=self.alternative).run_test(control=np.array(buckets_control),
                                                            test=np.array(buckets_test))

