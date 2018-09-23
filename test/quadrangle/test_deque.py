# -*- coding: utf-8 -*-

import unittest

from video699.quadrangle.deque import DequeMovingConvexQuadrangle
from video699.quadrangle.geos import GEOSConvexQuadrangle


QUADRANGLE_ID = 'quadrangle'


class TestDequeMovingConvexQuadrangle(unittest.TestCase):
    """Tests the ability of the DequeMovingConvexQuadrangle class to record quadrangle movement.

    """

    def setUp(self):
        self.first_quadrangle = GEOSConvexQuadrangle(
            top_left=(5, 3),
            top_right=(3, 5),
            bottom_left=(3, 1),
            bottom_right=(1, 3),
        )
        self.second_quadrangle = GEOSConvexQuadrangle(
            top_left=(4, 2),
            top_right=(6, 1),
            bottom_left=(6, 4),
            bottom_right=(8, 2),
        )
        self.third_quadrangle = GEOSConvexQuadrangle(
            top_left=(1, 4),
            top_right=(0, 4),
            bottom_left=(1, 2),
            bottom_right=(0, 2),
        )
        self.fourth_quadrangle = GEOSConvexQuadrangle(
            top_left=(0, 0),
            top_right=(10, 0),
            bottom_left=(0, 10),
            bottom_right=(10, 10),
        )

    def test_iteration(self):
        first_quadrangle = self.first_quadrangle
        second_quadrangle = self.second_quadrangle
        third_quadrangle = self.third_quadrangle

        moving_convex_quadrangle = DequeMovingConvexQuadrangle(QUADRANGLE_ID)
        moving_convex_quadrangle.add(first_quadrangle)
        moving_convex_quadrangle.add(second_quadrangle)
        moving_convex_quadrangle.add(third_quadrangle)

        forward_iterator = iter(moving_convex_quadrangle)
        self.assertEqual(
            (first_quadrangle, second_quadrangle, third_quadrangle),
            tuple(forward_iterator),
        )

        reversed_iterator = reversed(moving_convex_quadrangle)
        self.assertEqual(
            (third_quadrangle, second_quadrangle, first_quadrangle),
            tuple(reversed_iterator),
        )

    def test_window_size(self):
        first_quadrangle = self.first_quadrangle
        second_quadrangle = self.second_quadrangle
        third_quadrangle = self.third_quadrangle
        fourth_quadrangle = self.fourth_quadrangle

        moving_convex_quadrangle = DequeMovingConvexQuadrangle(QUADRANGLE_ID, window_size=3)
        moving_convex_quadrangle.add(first_quadrangle)
        moving_convex_quadrangle.add(second_quadrangle)
        moving_convex_quadrangle.add(third_quadrangle)
        moving_convex_quadrangle.add(fourth_quadrangle)

        earliest_quadrangle = next(iter(moving_convex_quadrangle))
        self.assertEqual(second_quadrangle, earliest_quadrangle)

        current_quadrangle = next(reversed(moving_convex_quadrangle))
        self.assertEqual(fourth_quadrangle, current_quadrangle)


if __name__ == '__main__':
    unittest.main()
