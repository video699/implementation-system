# -*- coding: utf-8 -*-

"""This module implements stubs for testing purposes.

"""

from .interface import VideoABC, FrameABC


class VideoStub(VideoABC):
    """A hypothetical video captured as a specified date, and time.

    Parameters
    ----------
    datetime : aware datetime
        The date, and time at which the video was captured.

    Attributes
    ----------
    fps : int or None
        The framerate of the video in frames per second. If None, then the framerate is unknown.
    width : int or None
        The width of the video. If None, then the width is unknown.
    height : int or None
        The height of the video. If None, then the height is unknown.
    datetime : aware datetime
        The date, and time at which the video was captured.
    """

    fps = None
    width = None
    height = None

    def __init__(self, datetime):
        self._datetime = datetime

    @property
    def datetime(self):
        return self._datetime

    def __iter__(self, datetime):
        """The hypothetical video registers no frames."""
        return ()


class FrameStub(FrameABC):
    """A hypothetical frame of a video.

    Parameters
    ----------
    video : VideoABC
        The video containing the frame.

    Attributes
    ----------
    video : VideoABC
        The video containing the frame.
    number : int or None
        The frame number, i.e. the position of the frame in the video. If None, then the frame
        number is unknown.
    image : array_like or None
        The image data of the frame. If None, then no image data for the frame are available.
    datetime : aware datetime
        The date, and time at which the frame was captured.
    """

    number = None
    image = None

    def __init__(self, video):
        self._video = video

    @property
    def video(self):
        return self._video
