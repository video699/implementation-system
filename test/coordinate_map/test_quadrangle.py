# -*- coding: utf-8 -*-

import os
import unittest

import cv2 as cv
from video699.coordinate_map.quadrangle import Quadrangle


FRAME_IMAGE_PATHNAME = os.path.join(
    os.path.dirname(__file__),
    'test_quadrangle',
    'sample_frame_image.png',
)


class TestQuadrangle(unittest.TestCase):
    """Tests the ability of the Quadrangle class to map image data in a frame coordinate system.

    """

    def setUp(self):
        self.frame_image = cv.imread(FRAME_IMAGE_PATHNAME)

    def test_red_screen(self):
        coordinate_map = Quadrangle(
            top_left=(50, 210),
            top_right=(30, 55),
            btm_left=(300, 250),
            btm_right=(300, 20),
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
        coordinate_map = Quadrangle(
            top_left=(95, 385),
            top_right=(560, 360),
            btm_left=(75, 440),
            btm_right=(570, 450),
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

    def test_green_blue(self):
        coordinate_map = Quadrangle(
            top_left=(462, 112),
            top_right=(580, 120),
            btm_left=(460, 300),
            btm_right=(600, 160),
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
