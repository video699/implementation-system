# -*- coding: utf-8 -*-

import unittest

from numpy.testing import assert_array_almost_equal
from video699.common import benjamini_hochberg


class TestBenjaminiHochberg(unittest.TestCase):
    """Tests the ability of the benjamini_hochberg method to compute adjusted p-values.

    """

    def test_empty(self):
        p_values = ()
        q_values = tuple(benjamini_hochberg(p_values))
        self.assertEqual((), q_values)

    def test_nonempty(self):
        p_values = (0.005, 0.009, 0.05, 0.1, 0.2, 0.3)
        q_values = tuple(benjamini_hochberg(p_values))
        assert_array_almost_equal((0.027, 0.027, 0.1, 0.15, 0.24, 0.3), q_values)


if __name__ == '__main__':
    unittest.main()
