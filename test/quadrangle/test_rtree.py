# -*- coding: utf-8 -*-

import unittest

from video699.quadrangle.geos import GEOSConvexQuadrangle
from video699.quadrangle.rtree import RTreeConvexQuadrangleIndex


FIRST_QUADRANGLE = GEOSConvexQuadrangle(
    top_left=(5, 3),
    top_right=(3, 5),
    bottom_left=(3, 1),
    bottom_right=(1, 3),
)
FIRST_QUADRANGLE_AREA = 8

SECOND_QUADRANGLE = GEOSConvexQuadrangle(
    top_left=(4, 2),
    top_right=(6, 1),
    bottom_left=(6, 4),
    bottom_right=(8, 2),
)
SECOND_QUADRANGLE_AREA = 6

THIRD_QUADRANGLE = GEOSConvexQuadrangle(
    top_left=(1, 4),
    top_right=(0, 4),
    bottom_left=(1, 2),
    bottom_right=(0, 2),
)
THIRD_QUADRANGLE_AREA = 1 * 2

FOURTH_QUADRANGLE = GEOSConvexQuadrangle(
    top_left=(0, 0),
    top_right=(10, 0),
    bottom_left=(0, 10),
    bottom_right=(10, 10),
)
FOURTH_QUADRANGLE_AREA = 10**2

FIFTH_QUADRANGLE = GEOSConvexQuadrangle(
    top_left=(0, 0),
    top_right=(3, 0),
    bottom_left=(0, 6),
    bottom_right=(3, 6),
)
FIFTH_QUADRANGLE_AREA = 3 * 6


class TestRTreeConvexQuadrangleIndex(unittest.TestCase):
    """Tests the ability of the RTreeConvexQuadrangleIndex class to retrieve convex rectangles.

    """

    def test_add(self):
        index = RTreeConvexQuadrangleIndex()
        index.add(FIRST_QUADRANGLE)
        index.add(SECOND_QUADRANGLE)

        self.assertEqual(2, len(index))
        self.assertEqual({FIRST_QUADRANGLE, SECOND_QUADRANGLE}, set(index.quadrangles))
        self.assertEqual(
            {
                FIRST_QUADRANGLE: FIRST_QUADRANGLE_AREA,
                SECOND_QUADRANGLE: SECOND_QUADRANGLE_AREA,
            },
            index.intersection_areas(FOURTH_QUADRANGLE),
        )

    def test_add_duplicates(self):
        index = RTreeConvexQuadrangleIndex()
        index.add(FIRST_QUADRANGLE)
        index.add(FIRST_QUADRANGLE)
        index.add(GEOSConvexQuadrangle(
            top_left=FIRST_QUADRANGLE.top_left,
            top_right=FIRST_QUADRANGLE.top_right,
            bottom_left=FIRST_QUADRANGLE.bottom_left,
            bottom_right=FIRST_QUADRANGLE.bottom_right,
        ))

        self.assertEqual(1, len(index))
        self.assertEqual({FIRST_QUADRANGLE}, set(index.quadrangles))
        self.assertEqual(
            {
                FIRST_QUADRANGLE: FIRST_QUADRANGLE_AREA,
            },
            index.intersection_areas(FOURTH_QUADRANGLE),
        )

    def test_discard(self):
        index = RTreeConvexQuadrangleIndex((FIRST_QUADRANGLE, SECOND_QUADRANGLE))
        index.discard(SECOND_QUADRANGLE)

        self.assertEqual(1, len(index))
        self.assertEqual({FIRST_QUADRANGLE}, set(index.quadrangles))
        self.assertEqual(
            {
                FIRST_QUADRANGLE: FIRST_QUADRANGLE_AREA,
            },
            index.intersection_areas(FOURTH_QUADRANGLE),
        )

    def test_discard_duplicate(self):
        index = RTreeConvexQuadrangleIndex((FIRST_QUADRANGLE, SECOND_QUADRANGLE))
        index.discard(GEOSConvexQuadrangle(
            top_left=SECOND_QUADRANGLE.top_left,
            top_right=SECOND_QUADRANGLE.top_right,
            bottom_left=SECOND_QUADRANGLE.bottom_left,
            bottom_right=SECOND_QUADRANGLE.bottom_right,
        ))

        self.assertEqual(1, len(index))
        self.assertEqual({FIRST_QUADRANGLE}, set(index.quadrangles))
        self.assertEqual(
            {
                FIRST_QUADRANGLE: FIRST_QUADRANGLE_AREA,
            },
            index.intersection_areas(FOURTH_QUADRANGLE),
        )

    def test_intersection_areas_of_disjoint_quadrangles(self):
        index = RTreeConvexQuadrangleIndex((FIRST_QUADRANGLE,))
        self.assertEqual({}, index.intersection_areas(SECOND_QUADRANGLE))

    def test_intersection_areas_of_touching_quadrangles(self):
        index = RTreeConvexQuadrangleIndex((FIRST_QUADRANGLE,))
        self.assertEqual({}, index.intersection_areas(THIRD_QUADRANGLE))

    def test_intersection_areas_of_crossing_quadrangles(self):
        index = RTreeConvexQuadrangleIndex((
            FIRST_QUADRANGLE,
            SECOND_QUADRANGLE,
            THIRD_QUADRANGLE,
            FOURTH_QUADRANGLE,
            FIFTH_QUADRANGLE,
        ))
        self.assertEqual({
            FIRST_QUADRANGLE: FIRST_QUADRANGLE_AREA / 2,
            THIRD_QUADRANGLE: THIRD_QUADRANGLE_AREA,
            FOURTH_QUADRANGLE: FIFTH_QUADRANGLE_AREA,
            FIFTH_QUADRANGLE: FIFTH_QUADRANGLE_AREA,
        }, index.intersection_areas(FIFTH_QUADRANGLE))


if __name__ == '__main__':
    unittest.main()
