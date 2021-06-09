import unittest

import cv2
import numpy as np
from matplotlib import pyplot as plt

from video699.screen.semantic_segmentation.fastai_detector import ALL_VIDEOS, FastAIScreenDetector
import warnings


class TestPostprocessing(unittest.TestCase):
    """
    Tests the ability of the AnnotatedScreenVideo class to detect its dimensions and produce frames.
    """
    warnings.simplefilter("ignore")

    @staticmethod
    def visualize_np_array(array):
        """
        Use in PyCharm IDE or any IDE, that plt.show() by default. Use for debugging test that does not pass. While
        inside breakpoint use evaluate expression and use `self.visualize_np_array(self.crossed_quadrangle)`
        """
        blank_image = np.zeros((576, 720))
        image = cv2.fillConvexPoly(blank_image, array, color=1)
        plt.imshow(image)
        plt.show()

    def setUp(self) -> None:
        self.frame = list(list(ALL_VIDEOS)[0])[0]
        self.detector = FastAIScreenDetector(debug=True)
        self.blank_image = np.zeros((576, 720))

        self.left_rectangle = np.array([[50, 50], [350, 50], [350, 300], [50, 300]])
        self.right_rectangle = np.array([[[360, 50], [700, 50], [700, 300], [360, 300]]])
        self.ratio_split_rectangle = np.array([[[50, 50], [700, 50], [700, 300], [50, 300]]])
        self.erosion_dilation_connection = np.array([[[350, 180], [360, 180], [360, 200], [350, 200]]])
        self.concave_quadrangle = np.array([[[50, 50], [300, 50], [150, 150], [50, 400]]])
        self.crossed_quadrangle = np.array([[[50, 50], [500, 50], [50, 500], [500, 500]]])
        self.baseline_methods = {'base': True, 'erosion_dilation': False, 'ratio_split': False,
                                 'base_lower_bound': 7, 'base_factors': [0.001, 0.01, 0.02, 0.05, 0.1, 0.5]}

        self.erosion_dilation_methods = {'base': False, 'erosion_dilation': True, 'ratio_split': False,
                                         'erosion_dilation_lower_bound': 7, 'erosion_dilation_factors': [0.1, 0.01],
                                         'erosion_dilation_kernel_size': 40}

        self.ratio_split_methods = {'base': True, 'erosion_dilation': False, 'ratio_split': True,
                                    'base_lower_bound': 7, 'base_factors': [0.01, 0.1, 0.5, 1.0, 1.5],
                                    'ratio_split_lower_bound': 0.7}

    def test_baseline_rectangles(self):
        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, **self.baseline_methods)
        self.assertEqual(len(screens), 0)

        # Add left rectangle
        cv2.fillConvexPoly(self.blank_image, self.left_rectangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, **self.baseline_methods)
        self.assertEqual(len(screens), 1)

        # Add right rectangle
        cv2.fillConvexPoly(self.blank_image, self.right_rectangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, **self.baseline_methods)
        self.assertEqual(len(screens), 2)

    def test_baseline_rectangles_connected(self):
        cv2.fillConvexPoly(self.blank_image, self.left_rectangle, color=1)
        cv2.fillConvexPoly(self.blank_image, self.right_rectangle, color=1)
        cv2.fillConvexPoly(self.blank_image, self.erosion_dilation_connection, color=1)

        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, **self.baseline_methods)
        self.assertEqual(len(screens), 1)

    def test_erosion_dilation_rectangles(self):
        screens = self.detector.post_processing(self.blank_image.astype('uint8'), self.frame, **self.baseline_methods)
        self.assertEqual(len(screens), 0)

        # Add left rectangle
        cv2.fillConvexPoly(self.blank_image, self.left_rectangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'),
                                                self.frame, **self.erosion_dilation_methods)
        self.assertEqual(len(screens), 1)

        # Add right rectangle
        cv2.fillConvexPoly(self.blank_image, self.right_rectangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'),
                                                self.frame, **self.erosion_dilation_methods)
        self.assertEqual(len(screens), 2)

    def test_erosion_dilation_rectangles_connected(self):
        cv2.fillConvexPoly(self.blank_image, self.left_rectangle, color=1)
        cv2.fillConvexPoly(self.blank_image, self.right_rectangle, color=1)
        cv2.fillConvexPoly(self.blank_image, self.erosion_dilation_connection, color=1)

        screens = self.detector.post_processing(self.blank_image.astype('uint8'),
                                                self.frame, **self.erosion_dilation_methods)
        self.assertEqual(len(screens), 2)

    def test_ratio_split_rectangles_invalid_methods(self):
        self.ratio_split_methods['base'] = False
        cv2.fillConvexPoly(self.blank_image, self.ratio_split_rectangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'),
                                                self.frame, **self.ratio_split_methods)
        self.assertEqual(len(screens), 0)

    def test_ratio_split_rectangles(self):
        cv2.fillConvexPoly(self.blank_image, self.ratio_split_rectangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'),
                                                self.frame, **self.ratio_split_methods)
        self.assertEqual(len(screens), 2)

    def test_retrieved_quadrangle_contour_convexity(self):
        cv2.fillConvexPoly(self.blank_image, self.concave_quadrangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'),
                                                self.frame, **self.baseline_methods)
        self.assertEqual(len(screens), 0)

    def test_crossed_quadrangle(self):
        cv2.fillConvexPoly(self.blank_image, self.crossed_quadrangle, color=1)
        screens = self.detector.post_processing(self.blank_image.astype('uint8'),
                                                self.frame, **self.baseline_methods)
        self.assertEqual(len(screens), 0)


if __name__ == '__main__':
    unittest.main()
