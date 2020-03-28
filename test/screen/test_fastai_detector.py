import unittest
from pathlib import Path

from video699.interface import ScreenDetectorABC
from video699.screen.semantic_segmentation.fastai_detector import FastAIScreenDetector


class TestFastAIScreenDetector(unittest.TestCase):
    """
    Tests the ability of the AnnotatedScreenVideo class to detect its dimensions and produce frames.
    """

    def setUp(self) -> None:
        self.detector = FastAIScreenDetector()

    def test_init(self):
        self.assertIsInstance(self.detector, ScreenDetectorABC)
        self.assertIsInstance(self.detector.model_path, Path)
        self.assertIsInstance(self.detector.labels_path, Path)
        self.assertIsInstance(self.detector.videos_path, Path)
        self.assertTrue(self.detector.model_path.parent.exists())
        self.assertTrue(self.detector.labels_path.exists())
        self.assertTrue(self.detector.videos_path.exists())

    def test_default_params(self):
        self.assertIsNotNone(self.detector.methods)
        self.assertIsInstance(self.detector.methods, dict)
        all_params = {'base', 'base_lower_bound', 'base_upper_bound', 'base_factors', 'erode_dilate',
                      'erode_dilate_lower_bound', 'erode_dilate_upper_bound', 'erode_dilate_factors',
                      'erode_dilate_iterations', 'ratio_split', 'ratio_split_lower_bound', 'ratio_split_upper_bound'}

        self.assertSetEqual(set(self.detector.methods.keys()), all_params)

        all_train_params = {'batch_size', 'resize_factor', 'frozen_epochs', 'unfrozen_epochs',
                            'frozen_lr', 'unfrozen_lr'}
        self.assertSetEqual(set(self.detector.train_params.keys()), all_train_params)



if __name__ == '__main__':
    unittest.main()
