# -*- coding: utf-8 -*-

"""This module defines interfaces, and abstract base classes.

The use of MAY, and MUST in the docstrings follows RFC 2119.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import timedelta
from functools import total_ordering


class VideoABC(ABC, Iterable):
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

        Note
        ----
        It MAY be possible to iterate repeatedly over all video frames.

        Returns
        -------
        frames : iterator of FrameABC
            An iterable of the frames of the video.
        """
        pass


@total_ordering
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

    def __hash__(self):
        return hash((self.video, self.number))

    def __eq__(self, other):
        if isinstance(other, FrameABC) and self.video == other.video:
            return self.frame_number == other.frame_number
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, FrameABC) and self.video == other.video:
            return self.frame_number < other.frame_number
        return NotImplemented


@total_ordering
class ConvexQuadrangleABC(ABC):
    """A convex quadrangle specifying a map between video frame and projection screen coordinates.

    Attributes
    ----------
    top_left : (scalar, scalar)
        The top left corner of the quadrangle in a video frame coordinate system.
    top_right : (scalar, scalar)
        The top right corner of the quadrangle in a video frame coordinate system.
    bottom_left : (scalar, scalar)
        The bottom left corner of the quadrangle in a video frame coordinate system.
    bottom_right : (scalar, scalar)
        The bottom right corner of the quadrangle in a video frame coordinate system.
    width : int
        The width of the screen in the screen coordinate space.
    height : int
        The height of the screen in the screen coordinate space.
    """

    @property
    @abstractmethod
    def top_left(self):
        pass

    @property
    @abstractmethod
    def top_right(self):
        pass

    @property
    @abstractmethod
    def bottom_left(self):
        pass

    @property
    @abstractmethod
    def bottom_right(self):
        pass

    @property
    @abstractmethod
    def width(self):
        pass

    @property
    @abstractmethod
    def height(self):
        pass

    @abstractmethod
    def intersection_area(self, other):
        """The area of the intersection of two convex quadrangles.

        Parameters
        ----------
        other : ConvexQuadrangleABC
            The other convex quadrangle.

        Returns
        -------
        intersection_area : scalar
            The area of the intersection of self, and the other convex quadrangle.
        """
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

    def __eq__(self, other):
        if isinstance(other, ConvexQuadrangleABC):
            return self.top_left == other.top_left \
                and self.top_right == other.top_right \
                and self.bottom_left == other.bottom_left \
                and self.bottom_right == other.bottom_right
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, ConvexQuadrangleABC):
            return self.top_left < other.top_left \
                and self.top_right < other.top_right \
                and self.bottom_left < other.bottom_left \
                and self.bottom_right < other.bottom_right
        return NotImplemented


class ScreenABC(ABC):
    """An abstract projection screen shown in a video frame.

    Attributes
    ----------
    frame : FrameABC
        A frame containing the projection screen.
    coordinates : ConvexQuadrangleABC
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


class DocumentABC(ABC, Iterable):
    """An abstract text document.

    Attributes
    ----------
    title : str or None
        The title of a document.
    author : str or None
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

        Note
        ----
        It MUST be possible to iterate repeatedly over all document pages.

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
    """

    @property
    @abstractmethod
    def document(self):
        pass

    @property
    @abstractmethod
    def number(self):
        pass

    @abstractmethod
    def image(self, width, height):
        """Returns the image data of the document page at the specified dimensions.

        Parameters
        ----------
        width : int
            The width of the image data.
        height : int
            The height of the image data.

        Returns
        -------
        image : array_like
            The image data of the page represented as an OpenCV CV_8UC3 BGR matrix.
        """
        pass
