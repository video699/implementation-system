# -*- coding: utf-8 -*-

"""This module implements splitting a video to scenes.

"""

from collections.abc import Iterator

import cv2 as cv
import numpy as np

from video699.configuration import get_configuration
from video699.interface import VideoABC


CONFIGURATION = get_configuration()['MeanSquaredErrorSceneDetector']


class MeanSquaredErrorSceneDetector(VideoABC, Iterator):
    def __init__(self, video):
        self._video = video
        self._iterable = self._read_video()

    @property
    def fps(self):
        return self._video._fps

    @property
    def width(self):
        return self._video._width

    @property
    def height(self):
        return self._video._height

    @property
    def datetime(self):
        return self._video._datetime

    @property
    def uri(self):
        return self._video._uri

    def __iter__(self):
        return self

    def _read_video(self):
        max_mse = CONFIGURATION.getfloat('max_mse')
        image_width = CONFIGURATION.getint('image_width')
        image_height = CONFIGURATION.getint('image_height')
        norm = 1.0 / 255
        previous_frame_image = None
        for current_frame in self._video:
            current_frame_image = current_frame.render(image_width, image_height)
            current_frame_image = cv.cvtColor(current_frame_image[:, :, :3], cv.COLOR_BGR2LAB)
            if previous_frame_image is None:
                previous_frame_image = current_frame_image
                yield current_frame
            else:
                mse = np.mean(
                    ((current_frame_image - previous_frame_image) * norm)**2.0
                )
                if mse > max_mse:
                    previous_frame_image = current_frame_image
                    yield current_frame

    def __next__(self):
        return next(self._iterable)
