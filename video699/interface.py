# -*- coding: utf-8 -*-

"""This module defines interfaces, and abstract base classes.

"""

from abc import ABC, abstractmethod
from datetime import timedelta


class VideoABC(ABC):
    """An abstract video.

    Attributes
    ----------
    fps : scalar
        The framerate of the video in frames per second.
    width : int
        The width of the video.
    height : int
        The height of the video.
    datetime : aware datetime
        The date, and time at which the video was captured.
    """

    @property
    @abstractmethod
    def fps(self):
        pass

    @property
    @abstractmethod
    def width(self):
        pass

    @property
    @abstractmethod
    def height(self):
        pass

    @property
    @abstractmethod
    def datetime(self):
        pass

    @abstractmethod
    def __iter__(self):
        """Produces an iterator of video frames.

        Returns
        -------
        frames : iterator of FrameABC
            An iterable of the frames of the video.
        """
        pass


class FrameABC(ABC):
    """An abstract frame of a video.

    Attributes
    ----------
    video : VideoABC
        The video containing the frame.
    number : int
        The frame number, i.e. the position of the frame in the video. Frame indexing is one-based,
        i.e. the first frame has number 1.
    image : array_like
        The image data of the frame represented as an OpenCV CV_8UC3 BGR matrix.
    width : int
        The width of the image data.
    height : int
        The height of the image data.
    datetime : aware datetime
        The date, and time at which the frame was captured.
    """

    @property
    @abstractmethod
    def video(self):
        pass

    @property
    @abstractmethod
    def number(self):
        pass

    @property
    @abstractmethod
    def image(self):
        pass

    @property
    def width(self):
        return self.video.width

    @property
    def height(self):
        return self.video.height

    @property
    def datetime(self):
        if self.video.fps is not None:
            return self.video.datetime + timedelta(seconds=(self.number - 1) / self.video.fps)
        return self.video.datetime


class CoordinateMapABC(ABC):
    """An abstract map between a video frame and projection screen coordinate systems.

    Attributes
    ----------
    width : int
        The width of the screen in the screen coordinate space.
    height : int
        The height of the screen in the screen coordinate space.
    """

    @property
    @abstractmethod
    def width(self):
        pass

    @property
    @abstractmethod
    def height(self):
        pass

    @abstractmethod
    def __call__(self, frame_image):
        """Maps image data in the frame coordinate system to the screen coordinate system.

        Parameters
        ----------
        frame_image : array_like
            Image data in the video frame coordinate system.

        Returns
        -------
        screen_image : array_like
            Image data in the projection screen coordinate system.
        """
        pass


class ScreenABC(ABC):
    """An abstract projection screen shown in a video frame.

    Attributes
    ----------
    frame : FrameABC
        A frame containing the projection screen.
    coordinates : CoordinateMapABC
        A map between frame and screen coordinates.
    image : array_like
        The image data of the projection screen represented as an OpenCV CV_8UC3 BGR matrix.
    width : int
        The width of the image data.
    height : int
        The height of the image data.
    """

    @property
    @abstractmethod
    def frame(self):
        pass

    @property
    @abstractmethod
    def coordinates(self):
        pass

    @property
    def image(self):
        return self.coordinates(self.frame.image)

    @property
    def width(self):
        return self.coordinates.width

    @property
    def height(self):
        return self.coordinates.height


class ScreenDetectorABC(ABC):
    """An abstract screen detector that maps video frames to lists of screens.

    """

    @abstractmethod
    def __call__(self, frame):
        """Converts a frame to an iterable of detected lit projection screens.

        Parameters
        ----------
        frame : FrameABC
            A frame of a video.

        Returns
        -------
        screens : iterable of ScreenABC
            An iterable of detected lit projection screens.
        """
        pass


class DocumentABC(ABC):
    """An abstract text document.

    Attributes
    ----------
    title : str
        The title of a document.

    author : str
        The author of a document.
    """

    @property
    @abstractmethod
    def title(self):
        pass

    @property
    @abstractmethod
    def author(self):
        pass

    @abstractmethod
    def __iter__(self):
        """Produces an iterator of document pages.

        Returns
        -------
        pages : iterator of PageABC
            An iterable of the pages of the document.
        """
        pass


class PageABC(ABC):
    """An abstract page of a document.

    Attributes
    ----------
    document : DocumentABC
        The document containing the page.
    number : int
        The page number, i.e. the position of the page in the document. Frame indexing is one-based,
        i.e. the first frame has number 1.
    image : array_like
        The image data of the page represented as an OpenCV CV_8UC3 BGR matrix.
    width : int
        The width of the image data.
    height : int
        The height of the image data.
    """

    @property
    @abstractmethod
    def document(self):
        pass

    @property
    @abstractmethod
    def number(self):
        pass

    @property
    @abstractmethod
    def image(self):
        pass

    @property
    @abstractmethod
    def width(self):
        pass

    @property
    @abstractmethod
    def height(self):
        pass
