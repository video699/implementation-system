# -*- coding: utf-8 -*-

from fractions import Fraction
import unittest

from numpy.testing import assert_array_almost_equal
from video699.common import (
    benjamini_hochberg,
    binomial_confidence_interval,
    change_aspect_ratio_by_upscaling,
    rescale_and_keep_aspect_ratio,
)


class TestBenjaminiHochberg(unittest.TestCase):
    """Tests the ability of the benjamini_hochberg function to compute adjusted p-values.

    """

    def test_empty(self):
        p_values = ()
        q_values = tuple(benjamini_hochberg(p_values))
        self.assertEqual((), q_values)

    def test_nonempty(self):
        p_values = (0.005, 0.009, 0.05, 0.1, 0.2, 0.3)
        q_values = tuple(benjamini_hochberg(p_values))
        assert_array_almost_equal((0.027, 0.027, 0.1, 0.15, 0.24, 0.3), q_values)


class TestBinomialConfidenceInterval(unittest.TestCase):
    """Tests the ability of the binomial_confidence_interval function to give confidence intervals.

    """

    def test_zero_trials(self):
        with self.assertRaises(ValueError):
            binomial_confidence_interval(num_successes=0, num_trials=0, significance_level=0.05)

    def test_more_successes_than_trials(self):
        with self.assertRaises(ValueError):
            binomial_confidence_interval(num_successes=10, num_trials=5, significance_level=0.05)

    def test_nonempty(self):
        pointwise_estimate, lower_bound, upper_bound = binomial_confidence_interval(
            num_successes=520,
            num_trials=1000,
            significance_level=0.05,
        )
        self.assertEqual(520 / 1000, pointwise_estimate)
        self.assertAlmostEqual(0.4890176, lower_bound)
        self.assertAlmostEqual(0.5508293, upper_bound)


class TestChangeAspectRatioByUpscaling(unittest.TestCase):
    """Tests the ability of the change_aspect_ratio_by_upscaling function to calculate dimensions.

    """

    def test_same_aspect_ratio(self):
        original_width, original_height = (800, 600)
        new_aspect_ratio = Fraction(4, 3)
        rescaled_width, rescaled_height = change_aspect_ratio_by_upscaling(
            original_width,
            original_height,
            new_aspect_ratio,
        )

        self.assertEqual(800, rescaled_width)
        self.assertEqual(600, rescaled_height)

    def test_wider_aspect_ratio(self):
        original_width, original_height = (800, 600)
        new_aspect_ratio = Fraction(16, 9)
        rescaled_width, rescaled_height = change_aspect_ratio_by_upscaling(
            original_width,
            original_height,
            new_aspect_ratio,
        )

        self.assertEqual(1067, rescaled_width)
        self.assertEqual(600, rescaled_height)

    def test_taller_aspect_ratio(self):
        original_width, original_height = (800, 600)
        new_aspect_ratio = Fraction(1, 1)
        rescaled_width, rescaled_height = change_aspect_ratio_by_upscaling(
            original_width,
            original_height,
            new_aspect_ratio,
        )

        self.assertEqual(800, rescaled_width)
        self.assertEqual(800, rescaled_height)

    def test_zero_dimensions(self):
        original_width, original_height = (800, 600)
        new_aspect_ratio = Fraction(0, 1)

        with self.assertRaises(ValueError):
            rescaled_width, rescaled_height = change_aspect_ratio_by_upscaling(
                original_width,
                original_height,
                new_aspect_ratio,
            )

        new_aspect_ratio = Fraction(800, 600)
        original_width, original_height = (0, 600)
        with self.assertRaises(ValueError):
            rescaled_width, rescaled_height = change_aspect_ratio_by_upscaling(
                original_width,
                original_height,
                new_aspect_ratio,
            )

        original_width, original_height = (800, 0)
        with self.assertRaises(ValueError):
            rescaled_width, rescaled_height = change_aspect_ratio_by_upscaling(
                original_width,
                original_height,
                new_aspect_ratio,
            )


class TestRescaleAndKeepAspectRatio(unittest.TestCase):
    """Tests the ability of the rescale_and_keep_aspect_ratio function to calculate dimensions.

    """

    def test_same_aspect_ratio(self):
        original_width, original_height = (800, 600)
        new_width, new_height = (1200, 900)
        rescaled_width, rescaled_height, top_margin, bottom_margin, left_margin, right_margin = \
            rescale_and_keep_aspect_ratio(original_width, original_height, new_width, new_height)

        self.assertEqual(1200, rescaled_width)
        self.assertEqual(900, rescaled_height)
        self.assertEqual(0, top_margin)
        self.assertEqual(0, bottom_margin)
        self.assertEqual(0, left_margin)
        self.assertEqual(0, right_margin)

    def test_wider_aspect_ratio(self):
        original_width, original_height = (800, 600)
        new_width, new_height = (960, 600)
        rescaled_width, rescaled_height, top_margin, bottom_margin, left_margin, right_margin = \
            rescale_and_keep_aspect_ratio(original_width, original_height, new_width, new_height)

        self.assertEqual(800, rescaled_width)
        self.assertEqual(600, rescaled_height)
        self.assertEqual(0, top_margin)
        self.assertEqual(0, bottom_margin)
        self.assertEqual(80, left_margin)
        self.assertEqual(80, right_margin)

    def test_taller_aspect_ratio(self):
        original_width, original_height = (800, 600)
        new_width, new_height = (800, 800)
        rescaled_width, rescaled_height, top_margin, bottom_margin, left_margin, right_margin = \
            rescale_and_keep_aspect_ratio(original_width, original_height, new_width, new_height)

        self.assertEqual(800, rescaled_width)
        self.assertEqual(600, rescaled_height)
        self.assertEqual(100, top_margin)
        self.assertEqual(100, bottom_margin)
        self.assertEqual(0, left_margin)
        self.assertEqual(0, right_margin)

    def test_zero_dimensions(self):
        original_width, original_height = (800, 600)
        new_width, new_height = (0, 600)
        with self.assertRaises(ValueError):
            rescale_and_keep_aspect_ratio(
                original_width,
                original_height,
                new_width,
                new_height,
            )

        new_width, new_height = (800, 0)
        with self.assertRaises(ValueError):
            rescale_and_keep_aspect_ratio(
                original_width,
                original_height,
                new_width,
                new_height,
            )

        original_width, original_height = (0, 600)
        new_width, new_height = (800, 600)
        with self.assertRaises(ValueError):
            rescale_and_keep_aspect_ratio(
                original_width,
                original_height,
                new_width,
                new_height,
            )

        original_width, original_height = (800, 0)
        with self.assertRaises(ValueError):
            rescale_and_keep_aspect_ratio(
                original_width,
                original_height,
                new_width,
                new_height,
            )

    def test_unspecified_new_width(self):
        original_width, original_height = (800, 600)
        new_width, new_height = (None, 900)
        rescaled_width, rescaled_height, top_margin, bottom_margin, left_margin, right_margin = \
            rescale_and_keep_aspect_ratio(original_width, original_height, new_width, new_height)

        self.assertEqual(1200, rescaled_width)
        self.assertEqual(900, rescaled_height)
        self.assertEqual(0, top_margin)
        self.assertEqual(0, bottom_margin)
        self.assertEqual(0, left_margin)
        self.assertEqual(0, right_margin)

    def test_unspecified_new_height(self):
        original_width, original_height = (800, 600)
        new_width, new_height = (1200, None)
        rescaled_width, rescaled_height, top_margin, bottom_margin, left_margin, right_margin = \
            rescale_and_keep_aspect_ratio(original_width, original_height, new_width, new_height)

        self.assertEqual(1200, rescaled_width)
        self.assertEqual(900, rescaled_height)
        self.assertEqual(0, top_margin)
        self.assertEqual(0, bottom_margin)
        self.assertEqual(0, left_margin)
        self.assertEqual(0, right_margin)

    def test_unspecified_new_dimensions(self):
        original_width, original_height = (800, 600)
        new_width, new_height = (None, None)
        rescaled_width, rescaled_height, top_margin, bottom_margin, left_margin, right_margin = \
            rescale_and_keep_aspect_ratio(original_width, original_height, new_width, new_height)

        self.assertEqual(800, rescaled_width)
        self.assertEqual(600, rescaled_height)
        self.assertEqual(0, top_margin)
        self.assertEqual(0, bottom_margin)
        self.assertEqual(0, left_margin)
        self.assertEqual(0, right_margin)


if __name__ == '__main__':
    unittest.main()
