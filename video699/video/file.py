# -*- coding: utf-8 -*-

"""This module implements reading a video from a video file.

"""

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import cv2 as cv

from video699.interface import VideoABC, FrameABC
from video699.frame.image import ImageFrame


class VideoFileFrame(FrameABC):
    """A frame of a video read from a video file.

    Parameters
    ----------
    video : VideoABC
        The video containing the frame.
    number : int
        The frame number, i.e. the position of the frame in the video. Frame indexing is one-based,
        i.e. the first frame has number 1.
    duration : timedelta
        The elapsed time since the beginning of the video.
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
    duration : timedelta
        The elapsed time since the beginning of the video.
    datetime : aware datetime
        The date, and time at which the frame was captured.
    """

    def __init__(self, video, number, duration, image):
        self._frame = ImageFrame(video, number, image)
        self._duration = duration

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
    def duration(self):
        return self._duration


class VideoFile(VideoABC, Iterator):
    """A video read from a video file.

    .. _RFC3987: https://tools.ietf.org/html/rfc3987

    Notes
    -----
    It is not possible to repeatedly iterate over all video frames.
    A video file is opened as soon as the class is instantiated, and released only after the
    finalization of the object or after the last frame has been read.

    Parameters
    ----------
    pathname : str
        The pathname of a video file.
    datetime : aware datetime
        The date, and time at which the video was captured.
    verbose : bool, optional
        Whether a progress bar will be shown during the reading of the video. False if unspecified.

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
    uri : string
        An IRI, as defined in RFC3987_, that uniquely indentifies the video over the entire lifetime
        of a program.

    Raises
    ------
    OSError
        If the video file cannot be opened by OpenCV.
    """

    def __init__(self, pathname, datetime, verbose=False):
        self._cap = cv.VideoCapture(pathname)
        self._iterable = self._read_video(pathname, verbose)
        self._fps = self._cap.get(cv.CAP_PROP_FPS)
        self._width = int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self._datetime = datetime
        self._uri = Path(pathname).resolve().as_uri()

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

    @property
    def uri(self):
        return self._uri

    def __iter__(self):
        return self

    def _read_video(self, pathname, verbose):
        if not self._cap.isOpened():
            raise OSError('Unable to open video file "{}"'.format(pathname))
        frame_number = 0
        first_frame_time = datetime.now()
        while True:
            video_duration = timedelta(milliseconds=self._cap.get(cv.CAP_PROP_POS_MSEC))
            frame_number += 1
            retval, bgr_frame_image = self._cap.read()
            if not retval:
                self._cap.release()
                if verbose:
                    print()
                break
            rgba_frame_image = cv.cvtColor(bgr_frame_image, cv.COLOR_BGR2RGBA)
            yield VideoFileFrame(self, frame_number, video_duration, rgba_frame_image)
            if verbose:
                last_frame_time = datetime.now()
                conversion_duration = last_frame_time - first_frame_time
                try:
                    conversion_speed = (
                        video_duration.total_seconds() /
                        conversion_duration.total_seconds()
                    )
                except ZeroDivisionError:
                    conversion_speed = 0
                status = '\rReading {}: frame {}, time {}, speed {:.2f}x'.format(
                    pathname,
                    frame_number,
                    video_duration,
                    conversion_speed,
                )
                print(status, end='')

    def __next__(self):
        return next(self._iterable)

    def __del__(self):
        self._cap.release()
