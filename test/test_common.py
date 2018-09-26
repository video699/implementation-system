# -*- coding: utf-8 -*-

from fractions import Fraction
import unittest

from numpy.testing import assert_array_almost_equal
from video699.common import (
    benjamini_hochberg,
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

    def test_zero_aspect_ratio(self):
        original_width, original_height = (800, 600)
        new_aspect_ratio = Fraction(0, 1)

        with self.assertRaises(ValueError):
            rescaled_width, rescaled_height = change_aspect_ratio_by_upscaling(
                original_width,
                original_height,
                new_aspect_ratio,
            )


if __name__ == '__main__':
    unittest.main()
