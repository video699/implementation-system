# -*- coding: utf-8 -*-

import os
import unittest

import cv2 as cv
from video699.convex_quadrangle import ConvexQuadrangle


FRAME_IMAGE_PATHNAME = os.path.join(
    os.path.dirname(__file__),
    'test_convex_quadrangle',
    'sample_frame_image.png',
)


class TestConvexQuadrangle(unittest.TestCase):
    """Tests the ability of the ConvexQuadrangle class to map image data in frame coordinate system.

    """

    def setUp(self):
        self.frame_image = cv.imread(FRAME_IMAGE_PATHNAME)

    def test_corner_coordinates(self):
        top_left = (0, 2)
        top_right = (2, 2)
        bottom_left = (0, 0)
        bottom_right = (2, 0)
        quadrangle = ConvexQuadrangle(top_left, top_right, bottom_left, bottom_right)
        self.assertEqual(top_left, quadrangle.top_left)
        self.assertEqual(top_right, quadrangle.top_right)
        self.assertEqual(bottom_left, quadrangle.bottom_left)
        self.assertEqual(bottom_right, quadrangle.bottom_right)

    def test_equality(self):
        first_quadrangle = ConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        second_quadrangle = ConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        third_quadrangle = ConvexQuadrangle(
            top_left=(-1, 1),
            top_right=(1, 1),
            bottom_left=(-1, -1),
            bottom_right=(1, -1),
        )
        self.assertTrue(first_quadrangle == second_quadrangle)
        self.assertFalse(first_quadrangle == third_quadrangle)

    def test_ordering(self):
        first_quadrangle = ConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        second_quadrangle = ConvexQuadrangle(
            top_left=(-1, 1),
            top_right=(1, 1),
            bottom_left=(-1, -1),
            bottom_right=(1, -1),
        )
        third_quadrangle = ConvexQuadrangle(
            top_left=(-2, 0),
            top_right=(0, 0),
            bottom_left=(-2, -2),
            bottom_right=(0, -2),
        )
        self.assertTrue(first_quadrangle > second_quadrangle)
        self.assertTrue(second_quadrangle > third_quadrangle)
        self.assertTrue(first_quadrangle > third_quadrangle)

    def test_intersection_area_of_equal_quadrangles(self):
        quadrangle = ConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        intersection_area = 4
        self.assertEqual(intersection_area, quadrangle.intersection_area(quadrangle))

    def test_intersection_area_of_crossing_quadrangles(self):
        first_quadrangle = ConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        second_quadrangle = ConvexQuadrangle(
            top_left=(-1, 1),
            top_right=(1, 1),
            bottom_left=(-1, -1),
            bottom_right=(1, -1),
        )
        intersection_area = 1
        self.assertEqual(intersection_area, first_quadrangle.intersection_area(second_quadrangle))
        self.assertEqual(intersection_area, second_quadrangle.intersection_area(first_quadrangle))

    def test_intersection_area_of_touching_quadrangles(self):
        first_quadrangle = ConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        second_quadrangle = ConvexQuadrangle(
            top_left=(-2, 0),
            top_right=(0, 0),
            bottom_left=(-2, -2),
            bottom_right=(0, -2),
        )
        intersection_area = 0
        self.assertEqual(intersection_area, first_quadrangle.intersection_area(second_quadrangle))
        self.assertEqual(intersection_area, second_quadrangle.intersection_area(first_quadrangle))

    def test_intersection_area_of_disjoint_quadrangles(self):
        first_quadrangle = ConvexQuadrangle(
            top_left=(0, 2),
            top_right=(2, 2),
            bottom_left=(0, 0),
            bottom_right=(2, 0),
        )
        second_quadrangle = ConvexQuadrangle(
            top_left=(-4, 0),
            top_right=(-2, 0),
            bottom_left=(-4, -4),
            bottom_right=(-2, -4),
        )
        intersection_area = 0
        self.assertEqual(intersection_area, first_quadrangle.intersection_area(second_quadrangle))
        self.assertEqual(intersection_area, second_quadrangle.intersection_area(first_quadrangle))

    def test_red_screen(self):
        coordinate_map = ConvexQuadrangle(
            top_left=(50, 210),
            top_right=(30, 55),
            bottom_left=(300, 250),
            bottom_right=(300, 20),
        )
        screen_image = coordinate_map(self.frame_image)
        height, width, _ = screen_image.shape
        self.assertTrue(height > width)

        blue, green, red = cv.split(screen_image)

        self.assertEqual(0, blue[0, 0])
        self.assertEqual(0, green[0, 0])
        self.assertEqual(255, red[0, 0])

        self.assertEqual(0, blue[0, width - 1])
        self.assertEqual(0, green[0, width - 1])
        self.assertEqual(255, red[0, width - 1])

        self.assertEqual(0, blue[height - 1, 0])
        self.assertEqual(0, green[height - 1, 0])
        self.assertEqual(255, red[height - 1, 0])

        self.assertEqual(0, blue[height - 1, width - 1])
        self.assertEqual(0, green[height - 1, width - 1])
        self.assertEqual(255, red[height - 1, width - 1])

        self.assertEqual(0, blue[int((height - 1) / 8), int((width - 1) / 2)])
        self.assertEqual(0, green[int((height - 1) / 8), int((width - 1) / 2)])
        self.assertEqual(0, red[int((height - 1) / 8), int((width - 1) / 2)])

    def test_green_screen(self):
        coordinate_map = ConvexQuadrangle(
            top_left=(95, 385),
            top_right=(560, 360),
            bottom_left=(75, 440),
            bottom_right=(570, 450),
        )
        screen_image = coordinate_map(self.frame_image)
        height, width, _ = screen_image.shape
        self.assertTrue(width > height)

        blue, green, red = cv.split(screen_image)

        self.assertEqual(0, blue[0, 0])
        self.assertEqual(255, green[0, 0])
        self.assertEqual(0, red[0, 0])

        self.assertEqual(0, blue[0, width - 1])
        self.assertEqual(255, green[0, width - 1])
        self.assertEqual(0, red[0, width - 1])

        self.assertEqual(0, blue[height - 1, 0])
        self.assertEqual(255, green[height - 1, 0])
        self.assertEqual(0, red[height - 1, 0])

        self.assertEqual(0, blue[height - 1, width - 1])
        self.assertEqual(255, green[height - 1, width - 1])
        self.assertEqual(0, red[height - 1, width - 1])

        self.assertEqual(0, blue[int((height - 1) / 2), (width - 1) - int((height - 1) / 4)])
        self.assertEqual(0, green[int((height - 1) / 2), (width - 1) - int((height - 1) / 4)])
        self.assertEqual(0, red[int((height - 1) / 2), (width - 1) - int((height - 1) / 4)])

    def test_blue_screen(self):
        coordinate_map = ConvexQuadrangle(
            top_left=(462, 112),
            top_right=(580, 120),
            bottom_left=(460, 300),
            bottom_right=(600, 160),
        )
        screen_image = coordinate_map(self.frame_image)
        height, width, _ = screen_image.shape
        self.assertTrue(width > height)

        blue, green, red = cv.split(screen_image)

        self.assertEqual(255, blue[0, 0])
        self.assertEqual(0, green[0, 0])
        self.assertEqual(0, red[0, 0])

        self.assertEqual(255, blue[0, width - 1])
        self.assertEqual(0, green[0, width - 1])
        self.assertEqual(0, red[0, width - 1])

        self.assertEqual(255, blue[height - 1, 0])
        self.assertEqual(0, green[height - 1, 0])
        self.assertEqual(0, red[height - 1, 0])

        self.assertEqual(255, blue[height - 1, width - 1])
        self.assertEqual(0, green[height - 1, width - 1])
        self.assertEqual(0, red[height - 1, width - 1])

        self.assertEqual(0, blue[int((height - 1) / 4), int((width - 1) / 4)])
        self.assertEqual(0, green[int((height - 1) / 4), int((width - 1) / 4)])
        self.assertEqual(0, red[int((height - 1) / 4), int((width - 1) / 4)])


if __name__ == '__main__':
    unittest.main()
