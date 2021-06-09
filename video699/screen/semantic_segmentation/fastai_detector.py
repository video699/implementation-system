# -*- coding: utf-8 -*-

"""
This module implements automatic detection and localization of projector screens on video using
semantic segmentation U-Net architecture implemented with FastAI library and post-processed into
ConvexQuadrangles
"""
import logging

import cv2
from fastai.vision.all import *
from typing import List

from video699.configuration import get_configuration
from video699.interface import ScreenABC, ScreenDetectorABC, FrameABC
from video699.screen.semantic_segmentation.common import parse_post_processing_params, NotFittedException, resize_pred
from video699.screen.semantic_segmentation.postprocessing import approximate
from video699.video.annotated import get_videos, AnnotatedSampledVideoScreenDetector

logging.captureWarnings(True)
LOGGER = logging.getLogger(__name__)

ALL_VIDEOS = set(get_videos().values())
CONFIGURATION = get_configuration()['FastAIScreenDetector']
VIDEOS_ROOT = Path(list(ALL_VIDEOS)[0].pathname).parents[2]
DEFAULT_VIDEO_PATH = VIDEOS_ROOT / 'video' / 'annotated'
DEFAULT_LABELS_PATH = VIDEOS_ROOT / 'screen' / 'labels'
DEFAULT_MODEL_PATH = VIDEOS_ROOT / 'screen' / 'models'


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
    src_shape : tuple
        A tuple consisting of height and width respectively.
    is_fitted : bool
        A flag to check is model is fitted already.
    self.learner : Learner
        A fastai model.
    """

    def __init__(self, debug=False, force=False):
        self.post_processing_params = parse_post_processing_params(CONFIGURATION)
        self.model_path = DEFAULT_MODEL_PATH
        self.labels_path = DEFAULT_LABELS_PATH
        self.videos_path = DEFAULT_VIDEO_PATH
        self.src_shape = np.array(
            [CONFIGURATION.getint('image_height'), CONFIGURATION.getint('image_width')])
        self.image_area = CONFIGURATION.getint('image_width') * CONFIGURATION.getint('image_height')
        self.learner = None
        self.is_fitted = False
        self.model_name = 'xresnet18_unet'
        self.init_model(CONFIGURATION.getint('batch_size'), size=CONFIGURATION.getint('size'))

        if not force:
            self.load()
        self.create_labels(force=True)
        self.train(CONFIGURATION.getint('epochs'))
        if not debug:
            self.save()

    def update_params(self, **kwargs):
        self.post_processing_params.update(
            {pair: kwargs[pair] for pair in kwargs if pair in self.post_processing_params.keys()})

    def init_model(self, batch_size, size):
        """
        Initialize learner with parameters set in constructor.
        """
        fnames = list(filter(lambda fname: 'frame' in str(fname), get_image_files(self.videos_path, recurse=True)))

        def label_func(x):
            return self.labels_path / x.parent.name / x.name

        codes = np.array(['non-screen', 'screen'])
        dls = SegmentationDataLoaders.from_label_func(self.model_path, fnames, label_func,
                                                      codes=codes,
                                                      bs=batch_size,
                                                      item_tfms=[Resize(size, method='squish')])
        self.learner = unet_learner(dls=dls,
                                    arch=models.xresnet.xresnet18,
                                    metrics=[Dice(), JaccardCoeff()],
                                    wd=1e-2).to_fp16()

    def train(self, epochs):
        """
        Train self.learner model.
        """
        if not self.is_fitted:
            self.learner.fine_tune(epochs)
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

        image = cv2.cvtColor(frame.image, cv2.COLOR_RGBA2RGB)
        with self.learner.no_bar():
            pred = self.learner.predict(image)
        pred = pred[0].numpy().astype('uint8')
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
            The detected screens in left-sorted order for single frame.
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

    def save(self):
        """
        Save the model into directory self.model_path/self.model_name.
        """
        if not self.is_fitted:
            raise NotFittedException
        self.learner.save(self.model_name)

    def load(self):
        """
        Load model from previously saved chunks.
        """

        try:
            self.learner.load(self.model_name)
            self.is_fitted = True
        except FileNotFoundError:
            LOGGER.info(f"Model not found in {self.model_path}/{self.model_name}.")

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
        shutil.rmtree(self.learner.path)

    def create_labels(self, force=False):
        """
        Create semantic segmentation binary mask in self.labels_path using ALL_VIDEOS.
        """
        actual_detector = AnnotatedSampledVideoScreenDetector()
        if not self.labels_path.exists() or force:
            self.labels_path.mkdir(parents=True, exist_ok=True)

        for video in ALL_VIDEOS:
            video_dir = self.labels_path / video.filename
            if not video_dir.exists() or force:
                video_dir.mkdir(parents=True, exist_ok=True)
            for frame in video:
                frame_path = video_dir / frame.filename
                if not frame_path.exists() or force:
                    mask = np.zeros((frame.height, frame.width, 3), dtype=np.uint8)
                    screens = actual_detector.detect(frame=frame)
                    for screen in screens:
                        points = (
                            screen.coordinates.top_left,
                            screen.coordinates.top_right,
                            screen.coordinates.bottom_right,
                            screen.coordinates.bottom_left
                        )
                        cv2.fillConvexPoly(mask,
                                           np.array([[[xi, yi]] for xi, yi in points]).astype(np.int32),
                                           (1, 1, 1))
                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                    cv2.imwrite(str(frame_path.absolute()), mask_gray)
