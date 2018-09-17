# -*- coding: utf-8 -*-

"""This module defines interfaces, and abstract base classes.

The use of MAY, and MUST in the docstrings follows RFC2119_.

.. _RFC2119: https://tools.ietf.org/html/rfc2119
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import timedelta
from functools import total_ordering


class EventABC(ABC):
    """An abstract event detected in a video.

    """

    @abstractmethod
    def write_xml(self, f):
        """Writes an XML representation of the event to a file-like object.

        Parameters
        ----------
        f : file-like object
            A writeable file-like object to which an XML representation of the event will be
            written.
        """
        pass

    def __repr__(self):
        return '<{classname}>'.format(
            classname=self.__class__.__name__,
        )


class EventDetectorABC(ABC):
    """An abstract event detector.

    """

    @abstractmethod
    def __iter__(self):
        """Produces an iterator of events.

        Note
        ----
        It MAY be possible to iterate repeatedly over all events.

        Returns
        -------
        frames : iterator of EventABC
            An iterable of events.
        """
        pass

    def __repr__(self):
        return '<{classname}>'.format(
            classname=self.__class__.__name__,
        )


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

    def __repr__(self):
        return '<{classname}, {width}x{height}px, {fps} fps, {datetime}>'.format(
            classname=self.__class__.__name__,
            width=self.width,
            height=self.height,
            fps=self.fps,
            datetime=self.datetime,
        )


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
        The image data of the frame as an OpenCV CV_8UC3 RGBA matrix, where the alpha channel (A)
        denotes the weight of a pixel. Fully transparent pixels, i.e. pixels with zero alpha, SHOULD
        be completely disregarded in subsequent computation.
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
        return self.video.datetime + timedelta(seconds=(self.number - 1) / self.video.fps)

    def __hash__(self):
        return hash((self.video, self.number))

    def __eq__(self, other):
        if isinstance(other, FrameABC) and self.video == other.video:
            return self.number == other.number
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, FrameABC) and self.video == other.video:
            return self.number < other.number
        return NotImplemented

    def __repr__(self):
        return '<{classname}, frame #{frame_number}, {width}x{height}px, {datetime}>'.format(
            classname=self.__class__.__name__,
            frame_number=self.number,
            width=self.width,
            height=self.height,
            datetime=self.datetime,
        )


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
    def transform(self, frame_image):
        """Transforms image data in the frame coordinate system to the screen coordinate system.

        Parameters
        ----------
        frame_image : array_like
            Image data in the video frame coordinate system as an OpenCV CV_8UC3 RGBA matrix, where
            the alpha channel (A) denotes the weight of a pixel. Fully transparent pixels, i.e.
            pixels with zero alpha, SHOULD be completely disregarded in subsequent computation.

        Returns
        -------
        screen_image : array_like
            Image data in the projection screen coordinate system as an OpenCV CV_8UC3 RGBA matrix,
            where the alpha channel (A) denotes the weight of a pixel. Fully transparent pixels,
            i.e. pixels with zero alpha, SHOULD be completely disregarded in subsequent computation.
            Pixels that originate from beyond the boundaries of the frame coordinate system MUST
            be fully transparent.
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

    def __repr__(self):
        return '<{classname}, {top_left}, {top_right}, {bottom_left}, {bottom_right}>'.format(
            classname=self.__class__.__name__,
            top_left=self.top_left,
            top_right=self.top_right,
            bottom_left=self.bottom_left,
            bottom_right=self.bottom_right,
        )


class ScreenABC(ABC):
    """An abstract projection screen shown in a video frame.

    Attributes
    ----------
    frame : FrameABC
        A video frame containing the projection screen.
    coordinates : ConvexQuadrangleABC
        A map between frame and screen coordinates.
    image : array_like
        The image data of the projection screen as an OpenCV CV_8UC3 RGBA matrix, where the alpha
        channel (A) denotes the weight of a pixel. Fully transparent pixels, i.e. pixels with zero
        alpha, SHOULD be completely disregarded in subsequent computation.
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
        return self.coordinates.transform(self.frame.image)

    @property
    def width(self):
        return self.coordinates.width

    @property
    def height(self):
        return self.coordinates.height

    def __repr__(self):
        return '<{classname}, {width}x{height}px, frame {frame} at {coordinates}>'.format(
            classname=self.__class__.__name__,
            width=self.width,
            height=self.heigth,
            frame=self.frame,
            coordinates=self.coordinates,
        )


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

    def __repr__(self):
        return '<{classname}>'.format(
            classname=self.__class__.__name__,
        )


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

    def __repr__(self):
        formatted_title = ', "{0}"'.format(self.title) if self.title is not None else ''
        formatted_author = ', "{0}"'.format(self.author) if self.author is not None else ''
        return '<{classname}{formatted_author}{formatted_title}>'.format(
            classname=self.__class__.__name__,
            formatted_author=formatted_author,
            formatted_title=formatted_title,
        )


class PageABC(ABC):
    """An abstract page of a document.

    Attributes
    ----------
    document : DocumentABC
        The document containing the page.
    number : int
        The page number, i.e. the position of the page in the document. Page indexing is one-based,
        i.e. the first page has number 1.
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
            The image data of the page as an OpenCV CV_8UC3 RGBA matrix, where the alpha channel (A)
            denotes the weight of a pixel. Fully transparent pixels, i.e. pixels with zero alpha,
            SHOULD be completely disregarded in subsequent computation. Any margins added to the
            image data, e.g. by keeping the aspect ratio of the page, MUST be fully transparent.
        """
        pass

    def __hash__(self):
        return hash((self.document, self.number))

    def __eq__(self, other):
        if isinstance(other, PageABC) and self.document == other.document:
            return self.number == other.number
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, PageABC) and self.document == other.document:
            return self.number < other.number
        return NotImplemented

    def __repr__(self):
        return '<{classname}, page #{page_number}>'.format(
            classname=self.__class__.__name__,
            page_number=self.number,
        )
