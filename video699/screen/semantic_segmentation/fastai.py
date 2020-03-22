# -*- coding: utf-8 -*-

"""This module implements automatic detection and localization of projector screens on video using
semantic segmentation U-Net architecture implemented with FastAI library and post-processed into
ConvexQuadrangles

"""
import logging
from functools import partial
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
from fastai.metrics import dice
from fastai.vision import load_learner, defaults, SegmentationLabelList, open_mask, \
    SegmentationItemList, \
    get_transforms, imagenet_stats, unet_learner, models

from video699.configuration import get_configuration
from video699.interface import (
    ScreenABC,
    ScreenDetectorABC
)
from video699.screen.semantic_segmentation.common import NotFittedException, acc, iou, \
    get_label_from_image_name, \
    parse_methods, cv_image_to_tensor, tensor_to_cv_binary_image, resize_pred, get_top_left_x, \
    create_labels, parse_lr
from video699.screen.semantic_segmentation.postprocessing import approximate
from video699.video.annotated import get_videos

logging.basicConfig(filename='example.log', level=logging.DEBUG)
LOGGER = getLogger(__name__)

ALL_VIDEOS = set(get_videos().values())
CONFIGURATION = get_configuration()['FastaiVideoScreenDetector']
VIDEOS_ROOT = Path(ALL_VIDEOS.pop().pathname).parents[2]
DEFAULT_VIDEO_PATH = VIDEOS_ROOT / 'video' / 'annotated'
DEFAULT_LABELS_PATH = VIDEOS_ROOT / 'screen' / 'labels'
DEFAULT_MODEL_PATH = VIDEOS_ROOT / 'screen' / 'models' / 'production.pkl'


class SegLabelListCustom(SegmentationLabelList):
    """
    Semantic segmentation custom label list that opens labels in binary mode. It is inherited from
    fastai class with RGB images.
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
                 videos_path=DEFAULT_VIDEO_PATH, methods=None, device='cpu'):
        defaults.device = torch.device(device)
        self.model_path = model_path
        self.labels_path = labels_path
        self.videos_path = videos_path
        self.src_shape = np.array(
            [CONFIGURATION.getint('image_width'), CONFIGURATION.getint('image_height')])

        if methods:
            self.methods = methods
        else:
            self.methods = parse_methods(CONFIGURATION)

        self.is_fitted = False

        try:
            # self.learner = self.load(filename=self.model_path)
            self.learner = load_learner(path=self.model_path.parent, file=self.model_path.name,
                                        bs=1)
            self.is_fitted = True
        except FileNotFoundError:
            LOGGER.warning(
                f"Learner was not found in path: {self.model_path}. New training initialized.")
            self.train()

    def save(self):
        pass

    def load(self, filename):
        pass

    def train(self):
        create_labels(videos=ALL_VIDEOS, labels_path=self.labels_path)
        batch_size = CONFIGURATION.getint('batch_size')
        size = self.src_shape // CONFIGURATION.getint('resize_factor')

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
        learn = unet_learner(data, models.resnet18, metrics=[acc, dice, iou])

        frozen_epochs = CONFIGURATION.getint('frozen_epochs')
        unfrozen_epochs = CONFIGURATION.getint('unfrozen_epochs')
        frozen_lr = parse_lr(CONFIGURATION['frozen_lr'])
        unfrozen_lr = parse_lr(CONFIGURATION['unfrozen_lr'])

        learn.fit_one_cycle(frozen_epochs, slice(frozen_lr))

        LOGGER.info("Unfreeze backbone part of the network.")
        learn.unfreeze()
        learn.fit_one_cycle(unfrozen_epochs, unfrozen_lr)
        learn.export(self.model_path)
        self.is_fitted = True

    def detect(self, frame, seg_debug=False):
        if not self.is_fitted:
            raise NotFittedException()

        # Semantic segmentation
        tensor = cv_image_to_tensor(frame.image)
        tensor = self.learner.predict(tensor)
        pred = tensor_to_cv_binary_image(tensor)
        resized = resize_pred(pred, tuple(self.src_shape))
        if seg_debug:
            return resized

        # Screen retrieval (Post processing)
        geos_quadrangles = approximate(resized, methods=self.methods)

        # Sort by top_left_x coordinate
        sorted_by_top_left_corner = sorted(geos_quadrangles, key=get_top_left_x)

        # Create screens (System Data Types)
        return [FastAIScreenDetectorVideoScreen(frame, screen_index, quadrangle) for
                screen_index, quadrangle in
                enumerate(sorted_by_top_left_corner)]
