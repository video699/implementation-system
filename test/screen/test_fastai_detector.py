import pathlib
import unittest

import cv2
import numpy as np

from video699.interface import ScreenDetectorABC
from video699.screen.semantic_segmentation.fastai_detector import FastAIScreenDetector, ALL_VIDEOS


class TestFastAIScreenDetector(unittest.TestCase):
    """
    Tests the ability of the AnnotatedScreenVideo class to detect its dimensions and produce frames.
    """

    def __init__(self, methodName):
        super().__init__(methodName)
        self.detector = FastAIScreenDetector(debug=True)

    def setUp(self) -> None:
        self.test_frame = list(ALL_VIDEOS.pop())[0]

    def test_init(self):
        self.assertIsInstance(self.detector, ScreenDetectorABC)
        self.assertIsInstance(self.detector, FastAIScreenDetector)
        self.assertIsInstance(self.detector.model_path, pathlib.Path)
        self.assertIsInstance(self.detector.labels_path, pathlib.Path)
        self.assertIsInstance(self.detector.videos_path, pathlib.Path)
        self.assertTrue(self.detector.labels_path.exists())
        self.assertTrue(self.detector.videos_path.exists())

    def test_default_params(self):
        self.assertIsNotNone(self.detector.post_processing_params)
        self.assertIsInstance(self.detector.post_processing_params, dict)
        all_post_processing_params = {'base', 'base_lower_bound', 'base_factors', 'erosion_dilation',
                                      'erosion_dilation_lower_bound', 'erosion_dilation_factors',
                                      'erosion_dilation_kernel_size', 'ratio_split', 'ratio_split_lower_bound'}
        self.assertSetEqual(set(self.detector.post_processing_params.keys()), all_post_processing_params)

    def test_save_load(self):
        self.detector.train(1)
        before_save = self.detector.semantic_segmentation(self.test_frame)
        self.detector.save()
        self.detector.load()
        self.detector.delete()
        after_save = self.detector.semantic_segmentation(self.test_frame)
        self.assertTrue(np.allclose(before_save, after_save, rtol=1e-05, atol=1e-08))

    def test_semantic_segmentation(self):
        pred = self.detector.semantic_segmentation(frame=self.test_frame)
        self.assertTrue(pred.shape[0] == self.test_frame.height)
        self.assertTrue(pred.shape[1] == self.test_frame.width)
        label = cv2.imread(str(
            self.detector.labels_path / pathlib.Path(self.test_frame.pathname).parent.name / self.test_frame.filename))
        label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
        intersection = np.logical_and(label, pred)
        union = np.logical_or(label, pred)
        iou_score = np.sum(intersection) / np.sum(union)
        self.assertTrue(iou_score > 0.95)


if __name__ == '__main__':
    unittest.main()
