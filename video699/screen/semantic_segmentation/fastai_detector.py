# -*- coding: utf-8 -*-

"""
This module implements automatic detection and localization of projector screens on video using
semantic segmentation U-Net architecture implemented with FastAI library and post-processed into
ConvexQuadrangles
"""
import io
import os
from functools import partial
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
from fastai.metrics import dice
from fastai.utils.mod_display import progress_disabled_ctx
from fastai.vision import load_learner, SegmentationLabelList, open_mask, \
    SegmentationItemList, get_transforms, imagenet_stats, unet_learner, models

from video699.configuration import get_configuration
from video699.interface import (
    ScreenABC,
    ScreenDetectorABC, FrameABC
)
from video699.screen.semantic_segmentation.common import NotFittedException, acc, get_label_from_image_name, \
    parse_post_processing_params, cv_image_to_tensor, tensor_to_cv_binary_image, resize_pred, create_labels, \
    iou_sem_seg, parse_train_params
from video699.screen.semantic_segmentation.postprocessing import approximate
from video699.video.annotated import get_videos
import warnings

warnings.filterwarnings('ignore')

# logging.basicConfig(filename='example.log', level=logging.WARNING)
LOGGER = getLogger(__name__)

ALL_VIDEOS = set(get_videos().values())
CONFIGURATION = get_configuration()['FastAIScreenDetector']
VIDEOS_ROOT = Path(list(ALL_VIDEOS)[0].pathname).parents[2]
DEFAULT_VIDEO_PATH = VIDEOS_ROOT / 'video' / 'annotated'
DEFAULT_LABELS_PATH = VIDEOS_ROOT / 'screen' / 'labels'
DEFAULT_MODEL_PATH = VIDEOS_ROOT / 'screen' / 'model' / 'model.pkl'
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    """A projection screen shown in a frame, detected by :class: FastAIScreenDetector.

    Parameters
    ----------
    frame : FrameABC
        A video frame containing the projection screen.
    coordinates : ConvexQuadrangleABC
        A map between frame and screen coordinates.
    screen_index : int
        An index of screen in the frame.

    Attributes
    ----------
    frame : FrameABC
        A video frame containing the projection screen.
    coordinates : ConvexQuadrangleABC
        A map between frame and screen coordinates.
    screen_index : int
        An index of screen in the frame.
    """

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
    """
    A projection screen shown in a frame, detected by :class: FastAIScreenDetector.

    Parameters
    ----------
    filtered_by : function
        A function used for filtering frames inside training videos.
    valid_func : function
        A function used for splitting frames into validation and training dataset.
    progressbar : bool
        A flag to turn on fastai training progressbar.

    Attributes
    ----------
    model_path : Path
        A path to the model to load and save.
    labels_path : Path
        A path to the labels to generate and load.
    videos_path : Path
        A path to the dataset videos.
    post_processing_params : dict
        A methods used for post_processing.
    train_params : dict
        A params used for training a model.
    filtered_by : function
        A function used for filtering frames inside training videos.
    valid_func : function
        A function used for splitting frames into validation and training dataset.
    device : string
        A device used for training : 'cpu' or 'cuda'
    progressbar : bool
        A flag to turn on fastai training progressbar.
    train : bool
        A flag to try the train the network if cannot be loaded by default where it is initialized. When off,
        initialization does not train but only load a model.
    src_shape : tuple
        A tuple consisting of height and width respectively.
    is_fitted : bool
        A flag to check is model is fitted already.
    self.learner : Learner
        A fastai model.
    """

    # noinspection PyTypeChecker
    def __init__(self, filtered_by: Callable = lambda fname: 'frame' in str(fname), valid_func: Callable = None,
                 progressbar: bool = True, train=True):

        self.post_processing_params = parse_post_processing_params(CONFIGURATION)
        self.train_params = parse_train_params(CONFIGURATION)
        self.progressbar = progressbar
        self.model_path = DEFAULT_MODEL_PATH
        self.labels_path = DEFAULT_LABELS_PATH
        self.videos_path = DEFAULT_VIDEO_PATH
        self.device = DEFAULT_DEVICE
        self.src_shape = np.array(
            [CONFIGURATION.getint('image_height'), CONFIGURATION.getint('image_width')])
        self.image_area = CONFIGURATION.getint('image_width') * CONFIGURATION.getint('image_height')

        self.filtered_by = filtered_by
        self.valid_func = valid_func
        self.learner = None
        self.is_fitted = False
        create_labels(videos=ALL_VIDEOS, labels_path=self.labels_path)
        self.init_model()
        try:
            self.load(self.model_path)
        except TypeError:
            LOGGER.info(f"Cannot load model from {self.model_path}. Training a new one.")
            if train:
                self.train()
                self.save()

    def init_model(self):
        """
        Initialize learner with parameters set in constructor.
        """

        size = self.src_shape // self.train_params['resize_factor']
        tfms = get_transforms(do_flip=True, flip_vert=False, max_lighting=0.8,
                              p_affine=0, p_lighting=0.5)

        tfms = (tfms[0][1:], tfms[1])

        get_label = partial(get_label_from_image_name, self.labels_path)

        if self.valid_func:
            src = (SegItemListCustom.from_folder(self.videos_path, ignore_empty=True, recurse=True)
                   .filter_by_func(self.filtered_by)
                   .split_by_valid_func(self.valid_func)
                   .label_from_func(get_label, classes=np.array(['non-screen', 'screen'])))

        else:
            src = (SegItemListCustom.from_folder(self.videos_path, ignore_empty=True, recurse=True)
                   .filter_by_func(self.filtered_by)
                   .split_none()
                   .label_from_func(get_label, classes=np.array(['non-screen', 'screen'])))

        LOGGER.info("Creating databunch with transformations")
        data = (src.transform(tfms, size=size, tfm_y=True)
                .databunch(bs=self.train_params['batch_size'])
                .normalize(imagenet_stats))

        LOGGER.info("Creating unet-learner with resnet18 backbone.")
        self.learner = unet_learner(data, models.resnet18, metrics=[acc, dice, iou_sem_seg])

    def train(self, **kwargs):
        """
        Train self.learner model.
        """
        self.train_params.update({pair: kwargs[pair] for pair in kwargs if pair in self.train_params.keys()})

        self.init_model()

        frozen_epochs = self.train_params['frozen_epochs']
        unfrozen_epochs = self.train_params['unfrozen_epochs']
        frozen_lr = self.train_params['frozen_lr']
        unfrozen_lr = self.train_params['unfrozen_lr']

        if self.progressbar:
            self.learner.fit_one_cycle(frozen_epochs, frozen_lr)
            self.learner.unfreeze()
            self.learner.fit_one_cycle(unfrozen_epochs, unfrozen_lr)

        else:
            with progress_disabled_ctx(self.learner) as self.learner:
                self.learner.fit_one_cycle(frozen_epochs, frozen_lr)
                self.learner.unfreeze()
                self.learner.fit_one_cycle(unfrozen_epochs, unfrozen_lr)

        self.is_fitted = True

    def detect(self, frame: FrameABC, **kwargs):
        """
        A screen detection: semantic segmentation and post-processing parts of algorithm merged in one function.
        Parameters
        ----------
        frame : FrameABC
            A frame from a video.
        kwargs : dict
            keyword parameters to rewrite default post-processing parameters.
        Returns
        -------
        screens: array-like
            A screens detected by FastAIScreenDetector.
        """
        if not self.is_fitted:
            raise NotFittedException()
        params_to_update = {pair: kwargs[pair] for pair in kwargs if pair in self.post_processing_params.keys()}
        self.post_processing_params.update(params_to_update)
        pred = self.semantic_segmentation(frame)
        screens = self.post_processing(pred, frame)
        return screens

    def semantic_segmentation(self, frame: FrameABC):
        """
        Semantic segmentation part of detecting the screens from frame
        Parameters
        ----------
        frame : FrameABC
            A single frame from a video.

        Returns
        -------
        pred: np.array
            A prediction from semantic segmentation for single frame.
        """
        if not self.is_fitted:
            raise NotFittedException

        tensor = cv_image_to_tensor(frame.image)
        tensor = self.learner.predict(tensor)
        pred = tensor_to_cv_binary_image(tensor)
        height, width = tuple(self.src_shape)
        resized = resize_pred(pred, width, height)
        return resized

    def post_processing(self, pred, frame: FrameABC, **kwargs):
        """
        A post-processing part of screen detection algorithm.
        Parameters
        ----------
        pred : np.array
            A prediction from semantic segmentation.
        frame : FrameABC
            A frame from a video.
        kwargs : dict
            Keyword arguments of different post-processing method. Not specified methods use default value. Optional

        Returns
        -------
        screens : array_like[FastAIScreenDetectorVideoScreen]
            The detecteed screens in left-sorted order for single frame.
        """
        if not self.is_fitted:
            raise NotFittedException

        params_to_update = {pair: kwargs[pair] for pair in kwargs if pair in self.post_processing_params.keys()}
        self.post_processing_params.update(params_to_update)

        geos_quadrangles = approximate(pred, post_processing_params=self.post_processing_params)
        sorted_by_top_left_corner = sorted(geos_quadrangles, key=lambda screen: screen.top_left[0])
        return [FastAIScreenDetectorVideoScreen(frame, screen_index, quadrangle) for
                screen_index, quadrangle in
                enumerate(sorted_by_top_left_corner)]

    def save(self, model_path: PathLike = None, chunk_size: int = 10000000):
        """
        Save the model into directory divided to multiple files.
        Parameters
        ----------
        model_path : Path
            A path to the model to load and save. Default set in __init__.
        chunk_size : int
            A size of chunks we divided model before saving.
        """
        if not self.is_fitted:
            raise NotFittedException

        if not model_path:
            model_path = self.model_path

        if not model_path.parent.exists():
            os.mkdir(model_path.parent.absolute())

        with io.BytesIO() as stream:
            self.learner.export(stream)
            stream.seek(0)
            part_number = 1
            chunk = stream.read(chunk_size)
            while chunk:
                part_name = model_path.parent / (str(model_path.stem) + str(part_number) + model_path.suffix)
                with open(part_name, mode='wb+') as chunk_file:
                    chunk_file.write(chunk)
                part_number += 1
                chunk = stream.read(chunk_size)

    def load(self, model_path: PathLike = None):
        """
        Load model from previously saved chunks.
        Parameters
        ----------
        model_path : Path
            A path to the model to load and save. Default set in __init__.
        """
        if not model_path:
            model_path = self.model_path
        part_number = 1
        chunks = []
        while (model_path.parent / (str(model_path.stem) + str(part_number) + model_path.suffix)).exists():
            with open(model_path.parent / (str(model_path.stem) + str(part_number) + model_path.suffix),
                      mode='rb') as chunk_file:
                chunks.append(chunk_file.read())
            part_number += 1

        with io.BytesIO(b"".join(chunks)) as stream:
            self.learner = load_learner(path=model_path.parent, file=stream, bs=1)

        self.is_fitted = True

    def semantic_segmentation_batch(self, frames: List[FrameABC]):
        """
        Semantic segmentation part of detecting the screens from multiple frames.
        Parameters
        ----------
        frames : array-like
            The frames from a video.

        Returns
        -------
        preds: np.array
            The semantic segmentation predictions for multiple frames.
        """
        return [self.semantic_segmentation(frame) for frame in frames]

    def post_processing_batch(self, preds, frames: List[FrameABC], **kwargs):
        """
        A post-processing part of screen detection algorithm for multiple predictions.
        Parameters
        ----------
        preds : np.array
            The predictions from semantic segmentation for multiple frames.
        frames : np.array
            The frames from a video.

        Returns
        -------
        screens : array_like
            The detected screens in left-sorted order for single frame for multiple predictions.
        """
        params_to_update = {pair: kwargs[pair] for pair in kwargs if pair in self.post_processing_params.keys()}
        self.post_processing_params.update(params_to_update)
        return [self.post_processing(preds, frames) for preds, frames in zip(preds, frames)]

    def detect_batch(self, frames: List[FrameABC], **kwargs):
        """
        A screen detection: semantic segmentation and post-processing parts of algorithm merged in one function for
        multiple frames.

        Parameters
        ----------
        frames : FrameABC
            The frames from a video.
        methods : dict
            A dictionary of different post-processing method. Not specified methods use default value. Optional

        Returns
        -------
        screens: array-like
            Screens detected by FastAIScreenDetector for multiple frames.
        """
        if not self.is_fitted:
            raise NotFittedException()
        params_to_update = {pair: kwargs[pair] for pair in kwargs if pair in self.post_processing_params.keys()}
        self.post_processing_params.update(params_to_update)
        preds = self.semantic_segmentation_batch(frames)
        screens = self.post_processing_batch(preds, frames)
        return screens

    def delete(self):
        """
        Delete saved model.
        """
        for file in os.listdir(str(self.model_path.parent)):
            os.remove(os.path.join(str(self.model_path.parent), file))
