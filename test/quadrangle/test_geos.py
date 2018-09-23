# -*- coding: utf-8 -*-

import os
import unittest

import cv2 as cv
from video699.quadrangle.geos import GEOSConvexQuadrangle


FRAME_IMAGE_PATHNAME = os.path.join(
    os.path.dirname(__file__),
    'test_geos',
    'sample_frame_image.png',
)


class TestGEOSConvexQuadrangle(unittest.TestCase):
    """Tests the ability of the GEOSConvexQuadrangle class to map images in frame coordinate systems.

    """

    def setUp(self):
        frame_image_bgr = cv.imread(FRAME_IMAGE_PATHNAME)
        self.frame_image = cv.cvtColor(frame_image_bgr, cv.COLOR_BGR2RGBA)

    def test_corner_coordinates(self):
        top_left = (0, 2)
        top_right = (2, 2)
        bottom_left = (0, 0)
        bottom_right = (2, 0)
        quadrangle = GEOSConvexQuadrangle(top_left, top_right, bottom_left, bottom_right)
        self.assertEqual(top_left, quadrangle.top_left)
        self.assertEqual(top_right, quadrangle.top_right)
        self.assertEqual(bottom_left, quadrangle.bottom_left)
        self.assertEqual(bottom_right, quadrangle.bottom_right)

    def test_bounding_box_coordinates(self):
        top_left = (0, 2)
        top_right = (2, 2)
        bottom_left = (0, 0)
        bottom_right = (2, 0)
        quadrangle = GEOSConvexQuadrangle(top_left, top_right, bottom_left, bottom_right)
        top_left_bound = bottom_left
        bottom_right_bound = top_right
        self.assertEqual(top_left_bound, quadrangle.top_left_bound)
        self.assertEqual(bottom_right_bound, quadrangle.bottom_right_bound)

        top_left = (1, 0)
        top_right = (0, 1)
        bottom_left = (1, 2)
        bottom_right = (2, 1)
        quadrangle = GEOSConvexQuadrangle(top_left, top_right, bottom_left, bottom_right)
        top_left_bound = (0, 0)
        bottom_right_bound = (2, 2)
        self.assertEqual(top_left_bound, quadrangle.top_left_bound)
        self.assertEqual(bottom_right_bound, quadrangle.bottom_right_bound)

    def test_equality(self):
        first_quadrangle = GEOSConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        second_quadrangle = GEOSConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        third_quadrangle = GEOSConvexQuadrangle(
            top_left=(-1, 1),
            top_right=(1, 1),
            bottom_left=(-1, -1),
            bottom_right=(1, -1),
        )
        self.assertTrue(first_quadrangle == second_quadrangle)
        self.assertFalse(first_quadrangle == third_quadrangle)

    def test_ordering(self):
        first_quadrangle = GEOSConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        second_quadrangle = GEOSConvexQuadrangle(
            top_left=(-1, 1),
            top_right=(1, 1),
            bottom_left=(-1, -1),
            bottom_right=(1, -1),
        )
        third_quadrangle = GEOSConvexQuadrangle(
            top_left=(-2, 0),
            top_right=(0, 0),
            bottom_left=(-2, -2),
            bottom_right=(0, -2),
        )
        self.assertTrue(first_quadrangle > second_quadrangle)
        self.assertTrue(second_quadrangle > third_quadrangle)
        self.assertTrue(first_quadrangle > third_quadrangle)

    def test_intersection_area_of_equal_quadrangles(self):
        quadrangle = GEOSConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        intersection_area = 4
        self.assertEqual(intersection_area, quadrangle.intersection_area(quadrangle))

    def test_intersection_area_of_crossing_quadrangles(self):
        first_quadrangle = GEOSConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        second_quadrangle = GEOSConvexQuadrangle(
            top_left=(-1, 1),
            top_right=(1, 1),
            bottom_left=(-1, -1),
            bottom_right=(1, -1),
        )
        intersection_area = 1
        self.assertEqual(intersection_area, first_quadrangle.intersection_area(second_quadrangle))
        self.assertEqual(intersection_area, second_quadrangle.intersection_area(first_quadrangle))

    def test_intersection_area_of_touching_quadrangles(self):
        first_quadrangle = GEOSConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        second_quadrangle = GEOSConvexQuadrangle(
            top_left=(-2, 0),
            top_right=(0, 0),
            bottom_left=(-2, -2),
            bottom_right=(0, -2),
        )
        intersection_area = 0
        self.assertEqual(intersection_area, first_quadrangle.intersection_area(second_quadrangle))
        self.assertEqual(intersection_area, second_quadrangle.intersection_area(first_quadrangle))

    def test_intersection_area_of_disjoint_quadrangles(self):
        first_quadrangle = GEOSConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        second_quadrangle = GEOSConvexQuadrangle(
            top_left=(-4, 0),
            top_right=(-2, 0),
            bottom_left=(-4, -4),
            bottom_right=(-2, -4),
        )
        intersection_area = 0
        self.assertEqual(intersection_area, first_quadrangle.intersection_area(second_quadrangle))
        self.assertEqual(intersection_area, second_quadrangle.intersection_area(first_quadrangle))

    def test_red_screen(self):
        coordinate_map = GEOSConvexQuadrangle(
            top_left=(50, 210),
            top_right=(30, 55),
            bottom_left=(300, 250),
            bottom_right=(300, 20),
        )
        screen_image = coordinate_map.transform(self.frame_image)
        height, width, _ = screen_image.shape
        self.assertTrue(height > width)

        red, green, blue, alpha = cv.split(screen_image)

        screen_corners = ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))
        for coordinates in screen_corners:
            self.assertEqual(0, blue[coordinates])
            self.assertEqual(0, green[coordinates])
            self.assertEqual(255, red[coordinates])
            self.assertEqual(255, alpha[coordinates])

        black_circle_coordinates = (int((height - 1) / 8), int((width - 1) / 2))
        self.assertEqual(0, blue[black_circle_coordinates])
        self.assertEqual(0, green[black_circle_coordinates])
        self.assertEqual(0, red[black_circle_coordinates])
        self.assertEqual(255, alpha[black_circle_coordinates])

    def test_green_screen(self):
        coordinate_map = GEOSConvexQuadrangle(
            top_left=(95, 385),
            top_right=(560, 360),
            bottom_left=(75, 440),
            bottom_right=(570, 450),
        )
        screen_image = coordinate_map.transform(self.frame_image)
        height, width, _ = screen_image.shape
        self.assertTrue(width > height)

        red, green, blue, alpha = cv.split(screen_image)

        screen_corners = ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))
        for coordinates in screen_corners:
            self.assertEqual(0, blue[coordinates])
            self.assertEqual(255, green[coordinates])
            self.assertEqual(0, red[coordinates])
            self.assertEqual(255, alpha[coordinates])

        black_circle_coordinates = (int((height - 1) / 2), (width - 1) - int((height - 1) / 4))
        self.assertEqual(0, blue[black_circle_coordinates])
        self.assertEqual(0, green[black_circle_coordinates])
        self.assertEqual(0, red[black_circle_coordinates])
        self.assertEqual(255, alpha[black_circle_coordinates])

    def test_blue_screen(self):
        coordinate_map = GEOSConvexQuadrangle(
            top_left=(462, 112),
            top_right=(580, 120),
            bottom_left=(460, 300),
            bottom_right=(600, 160),
        )
        screen_image = coordinate_map.transform(self.frame_image)
        height, width, _ = screen_image.shape
        self.assertTrue(width > height)

        red, green, blue, alpha = cv.split(screen_image)

        screen_corners = ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))
        for coordinates in screen_corners:
            self.assertEqual(255, blue[coordinates])
            self.assertEqual(0, green[coordinates])
            self.assertEqual(0, red[coordinates])
            self.assertEqual(255, alpha[coordinates])

        black_circle_coordinates = (int((height - 1) / 4), int((width - 1) / 4))
        self.assertEqual(0, blue[black_circle_coordinates])
        self.assertEqual(0, green[black_circle_coordinates])
        self.assertEqual(0, red[black_circle_coordinates])
        self.assertEqual(255, alpha[black_circle_coordinates])

    def test_out_of_bounds_screen(self):
        coordinate_map = GEOSConvexQuadrangle(
            top_left=(-50, 210),
            top_right=(-30, 55),
            bottom_left=(700, 250),
            bottom_right=(700, 20),
        )
        screen_image = coordinate_map.transform(self.frame_image)
        height, width, _ = screen_image.shape
        self.assertTrue(height > width)

        red, green, blue, alpha = cv.split(screen_image)

        screen_corners = ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))
        for coordinates in screen_corners:
            self.assertEqual(0, alpha[coordinates])

        red_rectangle_coordinates = (int((height - 1) / 2), int((width - 1) / 4))
        self.assertEqual(0, blue[red_rectangle_coordinates])
        self.assertEqual(0, green[red_rectangle_coordinates])
        self.assertEqual(255, red[red_rectangle_coordinates])
        self.assertEqual(255, alpha[red_rectangle_coordinates])

    def test_area(self):
        empty_quadrangle = GEOSConvexQuadrangle(
            top_left=(0, 0),
            top_right=(0, 0),
            bottom_left=(0, 0),
            bottom_right=(0, 0),
        )
        self.assertEqual(0, empty_quadrangle.area)

        rectangle = GEOSConvexQuadrangle(
            top_left=(5, 3),
            top_right=(3, 5),
            bottom_left=(3, 1),
            bottom_right=(1, 3),
        )
        self.assertEqual(8, rectangle.area)

        trapezoid = GEOSConvexQuadrangle(
            top_left=(4, 0),
            top_right=(8, 0),
            bottom_left=(5, 6),
            bottom_right=(6, 6),
        )
        self.assertEqual((4 + 1) * 6 / 2, trapezoid.area)


if __name__ == '__main__':
    unittest.main()
