# -*- coding: utf-8 -*-

import unittest

from video699.quadrangle.geos import GEOSConvexQuadrangle
from video699.quadrangle.rtree import RTreeConvexQuadrangleIndex


class TestRTreeConvexQuadrangleIndex(unittest.TestCase):
    """Tests the ability of the RTreeConvexQuadrangleIndex class to retrieve convex quadrangles.

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
        self.fifth_quadrangle = GEOSConvexQuadrangle(
            top_left=(0, 0),
            top_right=(3, 0),
            bottom_left=(0, 6),
            bottom_right=(3, 6),
        )

    def test_add(self):
        first_quadrangle = self.first_quadrangle
        second_quadrangle = self.second_quadrangle
        fourth_quadrangle = self.fourth_quadrangle

        index = RTreeConvexQuadrangleIndex()
        index.add(first_quadrangle)
        index.add(second_quadrangle)

        self.assertEqual(2, len(index))
        self.assertEqual({first_quadrangle, second_quadrangle}, set(index.quadrangles))
        self.assertEqual(
            {
                first_quadrangle: first_quadrangle.area,
                second_quadrangle: second_quadrangle.area,
            },
            index.intersection_areas(fourth_quadrangle),
        )

    def test_add_duplicates(self):
        first_quadrangle = self.first_quadrangle
        fourth_quadrangle = self.fourth_quadrangle

        index = RTreeConvexQuadrangleIndex()
        index.add(first_quadrangle)
        index.add(first_quadrangle)
        index.add(GEOSConvexQuadrangle(
            top_left=first_quadrangle.top_left,
            top_right=first_quadrangle.top_right,
            bottom_left=first_quadrangle.bottom_left,
            bottom_right=first_quadrangle.bottom_right,
        ))

        self.assertEqual(1, len(index))
        self.assertEqual({first_quadrangle}, set(index.quadrangles))
        self.assertEqual(
            {
                first_quadrangle: first_quadrangle.area,
            },
            index.intersection_areas(fourth_quadrangle),
        )

    def test_discard(self):
        first_quadrangle = self.first_quadrangle
        fourth_quadrangle = self.fourth_quadrangle

        index = RTreeConvexQuadrangleIndex((first_quadrangle, self.second_quadrangle))
        index.discard(self.second_quadrangle)

        self.assertEqual(1, len(index))
        self.assertEqual({first_quadrangle}, set(index.quadrangles))
        self.assertEqual(
            {
                first_quadrangle: first_quadrangle.area,
            },
            index.intersection_areas(fourth_quadrangle),
        )

    def test_discard_duplicate(self):
        first_quadrangle = self.first_quadrangle
        second_quadrangle = self.second_quadrangle
        fourth_quadrangle = self.fourth_quadrangle

        index = RTreeConvexQuadrangleIndex((first_quadrangle, second_quadrangle))
        index.discard(GEOSConvexQuadrangle(
            top_left=second_quadrangle.top_left,
            top_right=second_quadrangle.top_right,
            bottom_left=second_quadrangle.bottom_left,
            bottom_right=second_quadrangle.bottom_right,
        ))

        self.assertEqual(1, len(index))
        self.assertEqual({first_quadrangle}, set(index.quadrangles))
        self.assertEqual(
            {
                first_quadrangle: first_quadrangle.area,
            },
            index.intersection_areas(fourth_quadrangle),
        )

    def test_clear(self):
        first_quadrangle = self.first_quadrangle
        second_quadrangle = self.second_quadrangle
        fourth_quadrangle = self.fourth_quadrangle

        index = RTreeConvexQuadrangleIndex((first_quadrangle, second_quadrangle))
        index.clear()

        self.assertEqual(0, len(index))
        self.assertEqual({}, index.intersection_areas(fourth_quadrangle))

    def test_intersection_areas_of_disjoint_quadrangles(self):
        first_quadrangle = self.first_quadrangle
        second_quadrangle = self.second_quadrangle

        index = RTreeConvexQuadrangleIndex((first_quadrangle,))
        self.assertEqual({}, index.intersection_areas(second_quadrangle))

    def test_intersection_areas_of_touching_quadrangles(self):
        first_quadrangle = self.first_quadrangle
        third_quadrangle = self.third_quadrangle

        index = RTreeConvexQuadrangleIndex((first_quadrangle,))
        self.assertEqual({}, index.intersection_areas(third_quadrangle))

    def test_intersection_areas_of_crossing_quadrangles(self):
        first_quadrangle = self.first_quadrangle
        second_quadrangle = self.second_quadrangle
        third_quadrangle = self.third_quadrangle
        fourth_quadrangle = self.fourth_quadrangle
        fifth_quadrangle = self.fifth_quadrangle

        index = RTreeConvexQuadrangleIndex((
            first_quadrangle,
            second_quadrangle,
            third_quadrangle,
            fourth_quadrangle,
            fifth_quadrangle,
        ))
        self.assertEqual({
            first_quadrangle: first_quadrangle.area / 2,
            third_quadrangle: third_quadrangle.area,
            fourth_quadrangle: fifth_quadrangle.area,
            fifth_quadrangle: fifth_quadrangle.area,
        }, index.intersection_areas(fifth_quadrangle))


if __name__ == '__main__':
    unittest.main()
