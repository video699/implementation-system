import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from video699.interface import ScreenDetectorABC
from video699.screen.semantic_segmentation.common import create_labels
from video699.screen.semantic_segmentation.fastai_detector import FastAIScreenDetector, ALL_VIDEOS, \
    DEFAULT_LABELS_PATH, VIDEOS_ROOT


class TestFastAIScreenDetector(unittest.TestCase):
    """
    Tests the ability of the AnnotatedScreenVideo class to detect its dimensions and produce frames.
    """

    def __init__(self, methodName):
        super().__init__(methodName)
        create_labels(ALL_VIDEOS, DEFAULT_LABELS_PATH)

    def setUp(self) -> None:
        self.detector = FastAIScreenDetector(
            filtered_by=lambda name: 'frame002000' in str(name),
            progressbar=False,
            train=False
        )
        self.detector.train_params.update({'resize_factor': 8, 'unfrozen_epochs': 1, 'frozen_epochs': 1})
        self.detector.model_path = VIDEOS_ROOT.parent / 'test' / 'screen' / 'test_model' / 'model.plk'
        self.test_frame = list(ALL_VIDEOS.pop())[0]

    def test_init(self):
        self.assertIsInstance(self.detector, ScreenDetectorABC)
        self.assertIsInstance(self.detector, FastAIScreenDetector)
        self.assertIsInstance(self.detector.model_path, Path)
        self.assertIsInstance(self.detector.labels_path, Path)
        self.assertIsInstance(self.detector.videos_path, Path)
        self.assertTrue(self.detector.labels_path.exists())
        self.assertTrue(self.detector.videos_path.exists())

    def test_default_params(self):
        self.assertIsNotNone(self.detector.post_processing_params)
        self.assertIsInstance(self.detector.post_processing_params, dict)
        all_post_processing_params = {'base', 'base_lower_bound', 'base_factors', 'erosion_dilation',
                                      'erosion_dilation_lower_bound', 'erosion_dilation_factors',
                                      'erosion_dilation_kernel_size', 'ratio_split', 'ratio_split_lower_bound'}
        self.assertSetEqual(set(self.detector.post_processing_params.keys()), all_post_processing_params)

        all_train_params = {'batch_size', 'resize_factor', 'frozen_epochs', 'unfrozen_epochs',
                            'frozen_lr', 'unfrozen_lr'}
        self.assertSetEqual(set(self.detector.train_params.keys()), all_train_params)

    def test_train_params(self):
        self.detector.train(unfrozen_epochs=0, frozen_epochs=0, not_parameter=0)
        self.assertEqual(self.detector.train_params['unfrozen_epochs'], 0)
        self.assertEqual(self.detector.train_params['frozen_epochs'], 0)
        self.assertTrue('not_parameter' not in self.detector.train_params.keys())

    def test_save_load(self):
        self.detector.train()
        before_save = self.detector.semantic_segmentation(self.test_frame)
        self.detector.save()
        self.detector.load()
        self.detector.delete()
        after_save = self.detector.semantic_segmentation(self.test_frame)
        self.assertTrue(np.allclose(before_save, after_save, rtol=1e-05, atol=1e-08))

    def test_save_load_custom_dir(self):
        self.detector.train()
        tmp = NamedTemporaryFile()
        before_save = self.detector.semantic_segmentation(self.test_frame)
        self.detector.save(Path(tmp.name))
        self.detector.load(Path(tmp.name))
        after_save = self.detector.semantic_segmentation(self.test_frame)
        self.assertTrue(np.allclose(before_save, after_save, rtol=1e-05, atol=1e-08))

    def test_semantic_segmentation(self):
        pass

    def test_post_processing(self):
        pass


if __name__ == '__main__':
    unittest.main()
