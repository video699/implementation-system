# -*- coding: utf-8 -*-

"""This module implements a frame of a video represented by a NumPy matrix containing image data.

"""

from ..interface import FrameABC


class ImageFrame(FrameABC):
    """A frame of a video represented by a NumPy matrix containing image data.

    Parameters
    ----------
    video : VideoABC
        The video containing the frame.
    number : int
        The frame number, i.e. the position of the frame in the video. Frame
        indexing is one-based, i.e. the first frame has number 1.
    image : ndarray
        The image data of the frame represented as an OpenCV CV_8UC3 BGR matrix.

    Attributes
    ----------
    video : VideoABC
        The video containing the frame.
    number : int
        The frame number, i.e. the position of the frame in the video. Frame
        indexing is one-based, i.e. the first frame has number 1.
    image : ndarray
        The image data of the frame.
    width : int
        The width of the image data.
    height : int
        The height of the image data.
    datetime : aware datetime
        The date, and time at which the frame was captured.
    """

    def __init__(self, video, number, image):
        self._video = video
        self._number = number
        self._image = image

    @property
    def video(self):
        return self._video

    @property
    def number(self):
        return self._number

    @property
    def image(self):
        return self._image
