# -*- coding: utf-8 -*-

"""This module implements splitting a video to scenes.

"""

from collections.abc import Iterator

import numpy as np

from ..configuration import get_configuration
from ..interface import VideoABC


CONFIGURATION = get_configuration()['FrameImageDistanceSceneDetector']


class FrameImageDistanceSceneDetector(VideoABC, Iterator):
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
        max_mean_distance = CONFIGURATION.getfloat('max_mean_distance')
        previous_frame = None
        for current_frame in self._video:
            if previous_frame is None:
                previous_frame = current_frame
                yield current_frame
            else:
                mean_distance = np.mean(
                    np.ravel(
                        np.abs((current_frame.image - previous_frame.image) / 255.0)
                    )
                )
                if mean_distance > max_mean_distance:
                    previous_frame = current_frame
                    yield current_frame

    def __next__(self):
        return next(self._iterable)
