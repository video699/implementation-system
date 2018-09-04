# -*- coding: utf-8 -*-

"""This module implements reading a video from a video file.

"""

import cv2 as cv

from ..interface import VideoABC
from ..frame.image import ImageFrame


class VideoFile(VideoABC):
    """A video read from a video file.

    Note
    ----
    A video file is opened as soon as the class is instantiated, and released only after the
    finalization of the object or after the last frame has been read.

    Parameters
    ----------
    pathname : str
        The pathname of a video file.
    datetime : aware datetime
        The date, and time at which the video was captured.

    Attributes
    ----------
    fps : int
        The framerate of the video in frames per second.
    width : int
        The width of the video.
    height : int
        The height of the video.
    datetime : aware datetime
        The date, and time at which the video was captured.
    """

    def __init__(self, pathname, datetime):
        self._cap = cv.VideoCapture(pathname)
        if not self._cap.isOpened():
            raise OSError('Unable to open video file "{}"'.format(pathname))
        self._is_finished = False
        self._fps = self._cap.get(cv.CAP_PROP_FPS)
        self._frame_number = 0
        self._width = self._cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self._height = self._cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        self._datetime = datetime

    @property
    def fps(self):
        return self._fps

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def datetime(self):
        return self._datetime

    def __iter__(self):
        return self

    def __next__(self):
        if self._is_finished:
            raise StopIteration
        retval, frame_image = self._cap.read()
        if not retval:
            self._is_finished = True
            self._cap.release()
            raise StopIteration
        self._frame_number += 1
        return ImageFrame(self, self._frame_number, frame_image)

    def __del__(self):
        self._cap.release()
