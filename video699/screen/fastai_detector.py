# -*- coding: utf-8 -*-

"""This module implements automatic detection and localization of projector screens on video using semantic
segmentation U-Net architecture implemented with FastAI library and post-processed into ConvexQuadrangles

"""
import os
from functools import partial
from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
import torch
from fastai.metrics import dice
from fastai.vision import load_learner, defaults, SegmentationLabelList, open_mask, SegmentationItemList, \
    get_transforms, imagenet_stats, unet_learner, models

from video699.configuration import get_configuration
from video699.interface import (
    ScreenABC,
    ScreenDetectorABC
)
from video699.screen.common import NotFittedException, acc, IOU, get_label_from_image_name, parse_methods, \
    cv_image_to_tensor, tensor_to_cv_binary_image, resize_pred, get_top_left_x
from video699.screen.postprocessing import approximate
from video699.video.annotated import get_videos, AnnotatedSampledVideoScreenDetector

LOGGER = getLogger(__name__)
ALL_VIDEOS = set(get_videos().values())
CONFIGURATION = get_configuration()['FastaiVideoScreenDetector']
VIDEOS_ROOT = Path(ALL_VIDEOS.pop().pathname).parents[2]
DEFAULT_VIDEO_PATH = VIDEOS_ROOT / 'video' / 'annotated'
DEFAULT_LABELS_PATH = VIDEOS_ROOT / 'screen' / 'binary_masks'
DEFAULT_MODEL_PATH = VIDEOS_ROOT / 'screen' / 'model.pkl'

defaults.device = torch.device('cpu')


class SegLabelListCustom(SegmentationLabelList):
    """
    Semantic segmentation custom label list that opens labels in binary mode. It is inherited from fastai class with
    RGB images.
    """

    def open(self, fn): return open_mask(fn, div=False, convert_mode='L')


class SegItemListCustom(SegmentationItemList):
    """
    Semantic segmentation custom item list with binary labels.

    """
    _label_cls = SegLabelListCustom


class FastAIScreenDetectorVideoScreen(ScreenABC):
    def __init__(self, frame, screen_index, coordinates):
        self._frame = frame
        self._screen_index = screen_index
        self._coordinates = coordinates

    @property
    def frame(self):
        return self._frame

    @property
    def coordinates(self):
        return self._coordinates


class FastAIScreenDetector(ScreenDetectorABC):
    def __init__(self, model_path=DEFAULT_MODEL_PATH, labels_path=DEFAULT_LABELS_PATH,
                 videos_path=DEFAULT_VIDEO_PATH, methods=None):
        self.model_path = model_path
        self.labels_path = labels_path
        self.videos_path = videos_path
        if methods:
            self.methods = methods
        else:
            self.methods = parse_methods(CONFIGURATION)

        self.is_fitted = False

        try:
            self.learner = load_learner(path=self.model_path.parent, file=self.model_path.name, bs=4)
            self.is_fitted = True
        except FileNotFoundError:
            LOGGER.info(f"Learner was not found in path: {self.model_path}. New training initialized.")
            self.train()

    def train(self):
        self.create_labels(videos=ALL_VIDEOS)
        batch_size = CONFIGURATION.getint('batch_size')
        src_shape = np.array(CONFIGURATION.getint('image_height'), CONFIGURATION.getint('image_width'))
        size = src_shape // CONFIGURATION.getint('resize_factor')

        tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0,
                              max_zoom=1.1, max_lighting=0.2, max_warp=1,
                              p_affine=0, p_lighting=0.75)

        get_label = partial(get_label_from_image_name, self.labels_path)
        src = (SegItemListCustom.from_folder(self.videos_path, ignore_empty=True, recurse=True)
               .filter_by_func(lambda name: 'frame' in str(name))
               .split_none()
               .label_from_func(get_label, classes=np.array(['non-screen', 'screen'])))

        LOGGER.info("Creating databunch with transformations")
        data = (src.transform(tfms, size=size, tfm_y=True)
                .databunch(bs=batch_size)
                .normalize(imagenet_stats))

        LOGGER.info("Creating unet-learner with resnet18 backbone.")
        learn = unet_learner(data, models.resnet18, metrics=[acc, dice, IOU])
        lr = 1e-3
        learn.fit_one_cycle(1, slice(lr))

        # LOGGER.info("Unfreeze backbone part of the network.")
        # learn.unfreeze()
        # lrs = slice(1e-4, lr / 5)
        # learn.fit_one_cycle(7, lrs)  # 12

        learn.export(self.model_path)
        self.is_fitted = True

    def detect(self, frame):
        if not self.is_fitted:
            raise NotFittedException()

        # Semantic segmentation
        tensor = cv_image_to_tensor(frame.image)
        tensor = self.learner.predict(tensor)
        pred = tensor_to_cv_binary_image(tensor)
        resized = resize_pred(pred, (CONFIGURATION.getint('image_height'), CONFIGURATION.getint('image_width')))

        # Screen retrieval (Post processing)
        geos_quadrangles = approximate(resized, methods=self.methods)

        # Sort by top_left_x coordinate
        sorted_by_top_left_corner = sorted(geos_quadrangles, key=get_top_left_x)

        # Create screens (System Data Types)
        return [FastAIScreenDetectorVideoScreen(frame, screen_index, quadrangle) for screen_index, quadrangle in
                enumerate(sorted_by_top_left_corner)]

    def create_labels(self, videos):
        actual_detector = AnnotatedSampledVideoScreenDetector()
        if not self.labels_path.absolute().exists():
            os.mkdir(self.labels_path.absolute())

        for video in videos:
            video_dir = self.labels_path / video.filename
            if not video_dir.exists():
                os.mkdir(video_dir.absolute())

            for frame in video:
                frame_path = video_dir / frame.filename
                if not frame_path.exists():
                    mask = np.zeros((frame.height, frame.width, 3), dtype=np.uint8)
                    screens = actual_detector.detect(frame=frame)
                    for screen in screens:
                        points = (
                            screen.coordinates.top_left,
                            screen.coordinates.top_right,
                            screen.coordinates.bottom_right,
                            screen.coordinates.bottom_left
                        )
                        cv2.fillConvexPoly(mask, np.array([[[xi, yi]] for xi, yi in points]).astype(np.int32),
                                           (1, 1, 1))

                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                    cv2.imwrite(str(frame_path.absolute()), mask_gray)
