import unittest

import cv2
import numpy as np

from video699.screen.semantic_segmentation.fastai_detector import ALL_VIDEOS, FastAIScreenDetector


class TestPostprocessing(unittest.TestCase):
    """
    Tests the ability of the AnnotatedScreenVideo class to detect its dimensions and produce frames.
    """

    def setUp(self) -> None:
        self.frame = list(list(ALL_VIDEOS)[0])[0]
        self.detector = FastAIScreenDetector()
        self.blank_image = np.zeros((576, 720))
        self.left_rectangle = np.array([[50, 50], [350, 50], [350, 300], [50, 300]])
        self.right_rectangle = np.array([[[360, 50], [700, 50], [700, 300], [360, 300]]])
        self.ratio_split_rectangle = np.array([[[50, 50], [700, 50], [700, 300], [50, 300]]])
        self.erode_dilate_connection = np.array([[[350, 180], [360, 180], [360, 200], [350, 200]]])
        self.baseline_methods = {'base': True, 'erode_dilate': False, 'ratio_split': False,
                                 'base_lower_bound': 7, 'base_upper_bound': 50,
                                 'base_factors': [0.01, 0.1, 0.5, 1.0, 1.5]}

        self.erode_dilate_methods = {'base': False, 'erode_dilate': True, 'ratio_split': False,
                                     'erode_dilate_lower_bound': 7, 'erode_dilate_upper_bound': 50,
                                     'erode_dilate_factors': [0.1, 0.01], 'erode_dilate_iterations': 40}

        self.ratio_split_methods = {'base': True, 'erode_dilate': False, 'ratio_split': True,
                                    'base_lower_bound': 7, 'base_upper_bound': 50,
                                    'base_factors': [0.01, 0.1, 0.5, 1.0, 1.5],
                                    'ratio_split_lower_bound': 0.7, 'ratio_split_upper_bound': 1.5}

    def test_baseline_rectangles(self):
        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, self.baseline_methods)
        self.assertEqual(len(screens), 0)

        # Add left rectangle
        cv2.fillConvexPoly(self.blank_image, self.left_rectangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, self.baseline_methods)
        self.assertEqual(len(screens), 1)

        # Add right rectangle
        cv2.fillConvexPoly(self.blank_image, self.right_rectangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, self.baseline_methods)
        self.assertEqual(len(screens), 2)

    def test_baseline_rectangles_connected(self):
        cv2.fillConvexPoly(self.blank_image, self.left_rectangle, color=1)
        cv2.fillConvexPoly(self.blank_image, self.right_rectangle, color=1)
        cv2.fillConvexPoly(self.blank_image, self.erode_dilate_connection, color=1)

        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, self.baseline_methods)
        self.assertEqual(len(screens), 1)

    def test_erode_dilate_rectangles(self):
        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, self.baseline_methods)
        self.assertEqual(len(screens), 0)

        # Add left rectangle
        cv2.fillConvexPoly(self.blank_image, self.left_rectangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, self.erode_dilate_methods)
        self.assertEqual(len(screens), 1)

        # Add right rectangle
        cv2.fillConvexPoly(self.blank_image, self.right_rectangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, self.erode_dilate_methods)
        self.assertEqual(len(screens), 2)

    def test_erode_dilate_rectangles_connected(self):
        cv2.fillConvexPoly(self.blank_image, self.left_rectangle, color=1)
        cv2.fillConvexPoly(self.blank_image, self.right_rectangle, color=1)
        cv2.fillConvexPoly(self.blank_image, self.erode_dilate_connection, color=1)

        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, self.erode_dilate_methods)
        self.assertEqual(len(screens), 2)

    def test_ratio_split_rectangles_invalid_methods(self):
        self.ratio_split_methods['base'] = False
        cv2.fillConvexPoly(self.blank_image, self.ratio_split_rectangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, self.ratio_split_methods)
        self.assertEqual(len(screens), 0)

    def test_ratio_split_rectangles(self):
        cv2.fillConvexPoly(self.blank_image, self.ratio_split_rectangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, self.ratio_split_methods)
        self.assertEqual(len(screens), 2)


if __name__ == '__main__':
    unittest.main()
