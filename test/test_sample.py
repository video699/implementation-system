# -*- coding: utf-8 -*-

from .context import video699

import unittest


class SampleTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_tautology(self):
        self.assertTrue(video699)


if __name__ == '__main__':
    unittest.main()
