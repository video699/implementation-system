# import unittest
# from pathlib import Path
# from tempfile import NamedTemporaryFile
#
# from video699.screen.semantic_segmentation.common import create_labels
#
# from video699.interface import ScreenDetectorABC
# from video699.screen.semantic_segmentation.fastai_detector import FastAIScreenDetector, ALL_VIDEOS, \
#     DEFAULT_LABELS_PATH
# import numpy as np
#
#
# class TestFastAIScreenDetector(unittest.TestCase):
#     """
#     Tests the ability of the AnnotatedScreenVideo class to detect its dimensions and produce frames.
#     """
#
#     def __init__(self, methodName):
#         super().__init__(methodName)
#         create_labels(ALL_VIDEOS, DEFAULT_LABELS_PATH)
#
#     def setUp(self) -> None:
#         self.detector = FastAIScreenDetector(
#             filtered_by=lambda name: 'frame002000' in str(name),
#             train_params={'unfrozen_epochs': 1, 'frozen_epochs': 1})
#         self.test_frame = list(ALL_VIDEOS.pop())[0]
#
#     def test_init(self):
#         self.assertIsInstance(self.detector, ScreenDetectorABC)
#         self.assertIsInstance(self.detector.model_path, Path)
#         self.assertIsInstance(self.detector.labels_path, Path)
#         self.assertIsInstance(self.detector.videos_path, Path)
#         self.assertTrue(self.detector.model_path.parent.exists())
#         self.assertTrue(self.detector.labels_path.exists())
#         self.assertTrue(self.detector.videos_path.exists())
#
#     def test_init_params(self):
#         self.assertIsNotNone(self.detector.methods)
#         self.assertIsInstance(self.detector.methods, dict)
#         all_params = {'base', 'base_lower_bound', 'base_upper_bound', 'base_factors', 'erode_dilate',
#                       'erode_dilate_lower_bound', 'erode_dilate_upper_bound', 'erode_dilate_factors',
#                       'erode_dilate_iterations', 'ratio_split', 'ratio_split_lower_bound', 'ratio_split_upper_bound'}
#
#         self.assertSetEqual(set(self.detector.methods.keys()), all_params)
#
#         all_train_params = {'batch_size', 'resize_factor', 'frozen_epochs', 'unfrozen_epochs',
#                             'frozen_lr', 'unfrozen_lr'}
#         self.assertSetEqual(set(self.detector.train_params.keys()), all_train_params)
#
#     def test_init_params_custom(self):
#         self.detector = FastAIScreenDetector(train_params={'batch_size': 4})
#         all_params = {'base', 'base_lower_bound', 'base_upper_bound', 'base_factors', 'erode_dilate',
#                       'erode_dilate_lower_bound', 'erode_dilate_upper_bound', 'erode_dilate_factors',
#                       'erode_dilate_iterations', 'ratio_split', 'ratio_split_lower_bound', 'ratio_split_upper_bound'}
#
#         self.assertSetEqual(set(self.detector.methods.keys()), all_params)
#
#         all_train_params = {'batch_size', 'resize_factor', 'frozen_epochs', 'unfrozen_epochs',
#                             'frozen_lr', 'unfrozen_lr'}
#         self.assertSetEqual(set(self.detector.train_params.keys()), all_train_params)
#         self.assertEqual(self.detector.train_params['batch_size'], 4)
#
#     def test_save_load(self):
#         self.detector.train()
#         before_save = self.detector.semantic_segmentation(self.test_frame)
#         self.detector.save()
#         self.detector.load()
#         after_save = self.detector.semantic_segmentation(self.test_frame)
#         self.assertTrue(np.allclose(before_save, after_save, rtol=1e-05, atol=1e-08))
#
#     def test_save_load_custom_dir(self):
#         self.detector.train()
#         tmp = NamedTemporaryFile()
#         before_save = self.detector.semantic_segmentation(self.test_frame)
#         self.detector.save(Path(tmp.name))
#         self.detector.load(Path(tmp.name))
#         after_save = self.detector.semantic_segmentation(self.test_frame)
#         self.assertTrue(np.allclose(before_save, after_save, rtol=1e-05, atol=1e-08))
#
#     def test_semantic_segmentation(self):
#         pass
#
#     def test_post_processing(self):
#         pass
#
#
# if __name__ == '__main__':
#     unittest.main()
