# -*- coding: utf-8 -*-

import unittest

import numpy as np

from video699.event.pearson import RollingPearsonR


class TestRollingPearsonR(unittest.TestCase):
    """Tests the ability of the RollingPearsonR class to compute weighted Pearson's :math:`r`.

    """

    def test_empty(self):
        pearsonr = RollingPearsonR()

        correlation_coefficient, p_value = pearsonr.next(
            observations_x=np.array((), dtype=np.uint8),
            observations_y=np.array((), dtype=np.uint8),
            observation_weights=np.array((), dtype=np.uint8),
        )

        self.assertEqual(0, correlation_coefficient)
        self.assertEqual(0, p_value)

    def test_nonempty(self):
        pearsonr = RollingPearsonR()

        correlation_coefficient, p_value = pearsonr.next(
            observations_x=np.array((1, 2, 3), dtype=np.uint8),
            observations_y=np.array((3, 4, 5), dtype=np.uint8),
            observation_weights=np.array((1, 1, 1), dtype=np.uint8),
        )

        self.assertAlmostEqual(1.0, correlation_coefficient)
        self.assertAlmostEqual(0.0, p_value)

        correlation_coefficient, p_value = pearsonr.next(
            observations_x=np.array((3, 2, 1), dtype=np.uint8),
            observations_y=np.array((6, 7, 8), dtype=np.uint8),
            observation_weights=np.array((1, 1, 1), dtype=np.uint8),
        )

        self.assertAlmostEqual(0.0, correlation_coefficient)
        self.assertAlmostEqual(1.0, p_value)

        correlation_coefficient, p_value = pearsonr.next(
            observations_x=np.array((1, 2, 3), dtype=np.uint8),
            observations_y=np.array((3, 4, 5), dtype=np.uint8),
            observation_weights=np.array((1, 1, 1), dtype=np.uint8),
        )

        self.assertAlmostEqual(1/6, correlation_coefficient)
        self.assertAlmostEqual(0.668231, p_value)

    def test_weights(self):
        pearsonr = RollingPearsonR()

        correlation_coefficient, p_value = pearsonr.next(
            observations_x=np.array((1, 2, 3, 4), dtype=np.uint8),
            observations_y=np.array((3, 4, 5, 6), dtype=np.uint8),
            observation_weights=np.array((1, 1, 1, 0), dtype=np.uint8),
        )

        self.assertAlmostEqual(1.0, correlation_coefficient)
        self.assertAlmostEqual(0.0, p_value)

        correlation_coefficient, p_value = pearsonr.next(
            observations_x=np.array((3, 2, 1, 0), dtype=np.uint8),
            observations_y=np.array((6, 7, 8, 9), dtype=np.uint8),
            observation_weights=np.array((1, 1, 1, 0), dtype=np.uint8),
        )

        self.assertAlmostEqual(0.0, correlation_coefficient)
        self.assertAlmostEqual(1.0, p_value)

        correlation_coefficient, p_value = pearsonr.next(
            observations_x=np.array((1, 2, 3, 4), dtype=np.uint8),
            observations_y=np.array((3, 4, 5, 6), dtype=np.uint8),
            observation_weights=np.array((1, 1, 1, 0), dtype=np.uint8),
        )

        self.assertAlmostEqual(1/6, correlation_coefficient)
        self.assertAlmostEqual(0.668231, p_value)

    def test_empty_window(self):
        with self.assertRaises(ValueError):
            RollingPearsonR(window_size=0)

    def test_nonempty_window(self):
        pearsonr = RollingPearsonR(window_size=2)

        correlation_coefficient, p_value = pearsonr.next(
            observations_x=np.array((1, 2, 3), dtype=np.uint8),
            observations_y=np.array((3, 4, 5), dtype=np.uint8),
            observation_weights=np.array((1, 1, 1), dtype=np.uint8),
        )

        self.assertAlmostEqual(1.0, correlation_coefficient)
        self.assertAlmostEqual(0.0, p_value)

        correlation_coefficient, p_value = pearsonr.next(
            observations_x=np.array((3, 2, 1), dtype=np.uint8),
            observations_y=np.array((6, 7, 8), dtype=np.uint8),
            observation_weights=np.array((1, 1, 1), dtype=np.uint8),
        )

        self.assertAlmostEqual(0.0, correlation_coefficient)
        self.assertAlmostEqual(1.0, p_value)

        correlation_coefficient, p_value = pearsonr.next(
            observations_x=np.array((1, 2, 3), dtype=np.uint8),
            observations_y=np.array((3, 4, 5), dtype=np.uint8),
            observation_weights=np.array((1, 1, 1), dtype=np.uint8),
        )

        self.assertAlmostEqual(0.0, correlation_coefficient)
        self.assertAlmostEqual(1.0, p_value)

    def test_cummulative_error(self):
        pearsonr = RollingPearsonR(window_size=2)

        correlation_coefficient, p_value = pearsonr.next(
            observations_x=np.array((1, 2, 3), dtype=np.uint8),
            observations_y=np.array((3, 4, 5), dtype=np.uint8),
            observation_weights=np.array((1, 1, 1), dtype=np.uint8),
        )

        self.assertAlmostEqual(1.0, correlation_coefficient)
        self.assertAlmostEqual(0.0, p_value)

        for _ in range(10000):
            pearsonr.next(
                observations_x=np.array((3, 2, 1), dtype=np.uint8),
                observations_y=np.array((6, 7, 8), dtype=np.uint8),
                observation_weights=np.array((1, 1, 1), dtype=np.uint8),
            )

        correlation_coefficient, p_value = pearsonr.next(
            observations_x=np.array((1, 2, 3), dtype=np.uint8),
            observations_y=np.array((3, 4, 5), dtype=np.uint8),
            observation_weights=np.array((1, 1, 1), dtype=np.uint8),
        )

        self.assertAlmostEqual(0.0, correlation_coefficient)
        self.assertAlmostEqual(1.0, p_value)


if __name__ == '__main__':
    unittest.main()
