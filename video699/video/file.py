# -*- coding: utf-8 -*-

"""This module implements reading a video from a video file.

"""

from collections.abc import Iterator
from datetime import timedelta

import cv2 as cv

from ..interface import VideoABC, FrameABC
from ..frame.image import ImageFrame


class VideoFileFrame(FrameABC):
    """A frame of a video read from a video file.

    Parameters
    ----------
    video : VideoABC
        The video containing the frame.
    number : int
        The frame number, i.e. the position of the frame in the video. Frame indexing is one-based,
        i.e. the first frame has number 1.
    delta : float
        The number of milliseconds elapsed since the beginning of the video.
    image : array_like
        The image data of the frame as an OpenCV CV_8UC3 RGBA matrix, where the alpha channel (A)
        is currently unused and all pixels are fully opaque, i.e. they have the maximum alpha of
        255.

    Attributes
    ----------
    video : VideoABC
        The video containing the frame.
    number : int
        The frame number, i.e. the position of the frame in the video. Frame indexing is one-based,
        i.e. the first frame has number 1.
    image : array_like
        The image data of the frame as an OpenCV CV_8UC3 RGBA matrix, where the alpha channel (A)
        is currently unused and all pixels are fully opaque, i.e. they have the maximum alpha of
        255.
    width : int
        The width of the image data.
    height : int
        The height of the image data.
    datetime : aware datetime
        The date, and time at which the frame was captured.
    """

    def __init__(self, video, number, delta, image):
        self._frame = ImageFrame(video, number, image)
        self._delta = timedelta(milliseconds=delta)

    @property
    def video(self):
        return self._frame.video

    @property
    def number(self):
        return self._frame.number

    @property
    def image(self):
        return self._frame.image

    @property
    def datetime(self):
        return self.video.datetime + self._delta


class VideoFile(VideoABC, Iterator):
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

    Raises
    ------
    OSError
        If the video file cannot be opened by OpenCV.
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
        delta = self._cap.get(cv.CAP_PROP_POS_MSEC)
        retval, bgr_frame_image = self._cap.read()
        if not retval:
            self._is_finished = True
            self._cap.release()
            raise StopIteration
        rgba_frame_image = cv.cvtColor(bgr_frame_image, cv.COLOR_BGR2RGBA)
        self._frame_number += 1
        return VideoFileFrame(self, self._frame_number, delta, rgba_frame_image)

    def __del__(self):
        self._cap.release()
