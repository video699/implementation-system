# -*- coding: utf-8 -*-

"""This module defines interfaces, and abstract base classes.

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and
OPTIONAL in the docstrings of this module and modules importing this module are to be interpreted as
described in RFC2119_.

.. _RFC2119: https://tools.ietf.org/html/rfc2119
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, MutableSet, Sized
from datetime import timedelta
from functools import lru_cache, total_ordering

from .configuration import get_configuration


SCREENABC_CONFIGURATION = get_configuration()['ScreenABC']
SCREENABC_LRU_CACHE_MAXSIZE = SCREENABC_CONFIGURATION.getint('lru_cache_maxsize')


class EventABC(ABC):
    """An abstract event detected in a video.

    """

    @abstractmethod
    def write_xml(self, xf):
        """Writes an XML fragment that represents the event to an XML file.

        Parameters
        ----------
        xf : lxml.etree.xmlfile
            An XML file.
        """
        pass

    def __repr__(self):
        return '<{classname}>'.format(
            classname=self.__class__.__name__,
        )


class EventDetectorABC(ABC, Iterable):
    """An abstract detector of events in a video.

    Notes
    -----
    It MAY be possible to repeatedly iterate over all events detected in a video.

    Attributes
    ----------
    video : VideoABC
        The video in which the events are detected.
    """

    @property
    @abstractmethod
    def video(self):
        pass

    def write_xml(self, xf):
        """Writes an XML document that represents all detected event to an XML file.

        Notes
        -----
        After the first invocation of ``write_xml``, it MAY be possible to call ``write_xml`` again
        and to iterate over all detected events in the video.

        Parameters
        ----------
        xf : lxml.etree.xmlfile
            An XML file.
        """
        xf.write_declaration()
        with xf.element('events', attrib={'video-uri': self.video.uri}):
            for event in self:
                event.write_xml(xf)
                xf.flush()

    @abstractmethod
    def __iter__(self):
        """Produces an iterator of all events detected in a video.

        Notes
        -----
        After the first iteration, it MAY be possible to call ``write_xml`` and to iterate over all
        detected events in the video again.

        Returns
        -------
        frames : iterator of EventABC
            An iterable of events.
        """
        pass

    def __repr__(self):
        return '<{classname}, {video}>'.format(
            classname=self.__class__.__name__,
            video=self.video,
        )


class VideoABC(ABC, Iterable):
    """An abstract video.

    .. _RFC3987: https://tools.ietf.org/html/rfc3987

    Notes
    -----
    It MAY be possible to repeatedly iterate over all video frames.

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
    uri : string
        An IRI, as defined in RFC3987_, that uniquely indentifies the video over the entire lifetime
        of a program.
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

    @property
    @abstractmethod
    def uri(self):
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

    def __repr__(self):
        return '<{classname}, {uri}, {width}x{height}px, {fps} fps, {datetime}>'.format(
            classname=self.__class__.__name__,
            width=self.width,
            height=self.height,
            fps=self.fps,
            datetime=self.datetime,
            uri=self.uri,
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
    top_left_bound : (scalar, scalar)
        The top left corner of the minimal bounding box that bounds the quadrangle in a video frame
        coordinate system.
    top_right : (scalar, scalar)
        The top right corner of the quadrangle in a video frame coordinate system.
    bottom_left : (scalar, scalar)
        The bottom left corner of the quadrangle in a video frame coordinate system.
    bottom_right : (scalar, scalar)
        The bottom right corner of the quadrangle in a video frame coordinate system.
    bottom_right_bound : (scalar, scalar)
        The bottom right corner of the minimal bounding box that bounds the quadrangle in a video
        frame coordinate system.
    width : int
        The width of the quadrangle in the screen coordinate space.
    height : int
        The height of the quadrangle in the screen coordinate space.
    area : scalar
        The area of the screen in the video frame coordinate system.
    """

    @property
    @abstractmethod
    def top_left(self):
        pass

    @property
    @abstractmethod
    def top_left_bound(self):
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
    def bottom_right_bound(self):
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
    def area(self):
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

    def __hash__(self):
        return hash((self.top_left, self.top_right, self.bottom_left, self.bottom_right))

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


class ConvexQuadrangleIndexABC(MutableSet):
    """An abstract index for the retrieval of convex quadrangles.

    Notes
    -----
    All requirements for subclassing :class:`collections.abc.MutableSet` (such as the requirement
    for a special constructor signature) apply. Mixin methods of :class:`MutableSet` (such as
    ``__ior__``, and ``__iand__``) that are not overriden here MAY be efficiently implemented by
    subclasses.

    Attributes
    ----------
    quadrangles : read-only set-like object of ConvexQuadrangleABC
        The convex quadrangles in the index.
    """

    @property
    @abstractmethod
    def quadrangles(self):
        pass

    @abstractmethod
    def add(self, quadrangle):
        pass

    @abstractmethod
    def discard(self, quadrangle):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def intersection_areas(self, input_quadrangle):
        """Retrieves convex quadrangles from the index that intersect an input convex quadrangle.

        Parameters
        ----------
        quadrangle : ConvexQuadrangleABC
            An input convex quadrangle.

        Returns
        -------
        quadrangles : dict of (ConvexQuadrangleABC, scalar)
            A map between convex quadrangles from the index that intersect the input convex
            quadrangle, and the intersection areas. All intersection areas MUST be greater than
            zero.
        """
        pass

    def __contains__(self, quadrangle):
        return quadrangle in self.quadrangles

    def __iter__(self):
        return iter(self.quadrangles)

    def __len__(self):
        return len(self.quadrangles)


class MovingConvexQuadrangleABC(ABC, Iterable):
    """An abstract convex quadrangle that moves in time.

    Notes
    -----
    It MUST be possible to repeatedly iterate over all tracked convex quadrangles.

    Attributes
    ----------
    current_quadrangle : ConvexQuadrangleABC
        The latest coordinates of the moving convex quadrangle.
    """

    @property
    def current_quadrangle(self):
        return next(reversed(self))

    def add(self, quadrangle):
        """Adds the movement of the convex quadrangle at the following time frame.

        Parameters
        ----------
        quadrangle : ConvexQuadrangleABC
            The corners of the moving convex quadrangle at the following time frame.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """Produces an iterator of the convex quadrangle movements from the past to the present.

        Returns
        -------
        quadrangles : iterator of ConvexQuadrangleABC
            The corners of the moving convex quadrangle from the earliest time frame to the current
            time frame.
        """
        pass

    @abstractmethod
    def __reversed__(self):
        """Produces an iterator of the convex quadrangle movements from the present to the past.

        Returns
        -------
        quadrangles : iterator of ConvexQuadrangleABC
            The corners of the moving convex quadrangle from the current time frame to the earliest
            time frame.
        """
        pass

    def __hash__(self):
        return hash(self.current_quadrangle)

    def __eq__(self, other):
        if isinstance(other, MovingConvexQuadrangleABC):
            return self.current_quadrangle == other.current_quadrangle
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, MovingConvexQuadrangleABC):
            return self.current_quadrangle < other.current_quadrangle
        return NotImplemented

    def __repr__(self):
        return '<{classname}, {current_quadrangle}>'.format(
            classname=self.__class__.__name__,
            current_quadrangle=self.current_quadrangle,
        )


class ConvexQuadrangleTrackerABC(ABC, Iterable, Sized):
    """An abstract tracker of the movement of convex quadrangles over time.

    Notes
    -----
    It MUST be possible to repeatedly iterate over all tracked convex quadrangles.

    """

    @abstractmethod
    def update(self, quadrangles):
        """Records convex quadrangles that exist in the current time frame.

        Parameters
        ----------
        current_quadrangles : iterable of ConvexQuadrangleABC
            The convex quadrangles in the current time frame.

        Returns
        -------
        appeared_quadrangles : set of MovingConvexQuadrangleABC
            The moving convex quadrangles that did not exist in the previous time frame and exist in
            the current time frame.
        existing_quadrangles : set of MovingConvexQuadrangleABC
            The moving convex quadrangles that existed in the previous time frame and exist in the
            current time frame.
        disappeared_quadrangles : set of MovingConvexQuadrangleABC
            The moving convex quadrangles that existed in the previous time frame and do not exist
            in the current time frame.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """Produces an iterator of the tracked convex quadrangles.

        Returns
        -------
        tracked_quadrangles : iterator of TrackedConvexQuadrangleABC
            An iterable of the tracked convex quadrangles.
        """
        pass

    @abstractmethod
    def __len__(self):
        """Produces the number of the tracked convex quadrangles.

        Returns
        -------
        length : int
            The number of the tracked convex quadrangles.
        """
        pass

    def __repr__(self):
        return '<{classname}, {length} tracked moving quadrangles>'.format(
            classname=self.__class__.__name__,
            length=len(self),
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
    @lru_cache(maxsize=SCREENABC_LRU_CACHE_MAXSIZE, typed=False)
    def image(self):
        return self.coordinates.transform(self.frame.image)

    @property
    def width(self):
        return self.coordinates.width

    @property
    def height(self):
        return self.coordinates.height

    def __hash__(self):
        return hash((self.frame, self.coordinates))

    def __eq__(self, other):
        if isinstance(other, ScreenABC):
            return self.frame == other.frame and self.coordinates == other.coordinates
        return NotImplemented

    def __repr__(self):
        return '<{classname}, {width}x{height}px, frame {frame} at {coordinates}>'.format(
            classname=self.__class__.__name__,
            width=self.width,
            height=self.height,
            frame=self.frame,
            coordinates=self.coordinates,
        )


class ScreenDetectorABC(ABC):
    """An abstract screen detector that maps video frames to lists of screens.

    """

    @abstractmethod
    def detect(self, frame):
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

    .. _RFC3987: https://tools.ietf.org/html/rfc3987

    Notes
    -----
    It MUST be possible to repeatedly iterate over all document pages.

    Attributes
    ----------
    title : str or None
        The title of a document.
    author : str or None
        The author of a document.
    uri : string
        An IRI, as defined in RFC3987_, that uniquely indentifies the document over the entire
        lifetime of a program.
    """

    @property
    @abstractmethod
    def title(self):
        pass

    @property
    @abstractmethod
    def author(self):
        pass

    @property
    @abstractmethod
    def uri(self):
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

    def __repr__(self):
        formatted_title = ', "{0}"'.format(self.title) if self.title is not None else ''
        formatted_author = ', "{0}"'.format(self.author) if self.author is not None else ''
        return '<{classname}, {uri}{formatted_author}{formatted_title}>'.format(
            classname=self.__class__.__name__,
            formatted_author=formatted_author,
            formatted_title=formatted_title,
            uri=self.uri,
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


class PageDetectorABC(ABC):
    """An abstract detector of document pages in projection screens.

    """

    def detect(self, appeared_screens, existing_screens, disappeared_screens):
        """Detects document pages in projection screens from a current video frame.

        Parameters
        ----------
        appeared_screens : iterator of ScreenABC
            Projection screens that did not exist in the previous video frame and exist in the
            current video frame.
        existing_screens : iterator of ScreenABC
            Projection screens that existed in the previous video frame and exist in the current
            video frame.
        disappeared_screens : iterator of ScreenABC
            Projection screens that existed in the previous video frame and do not exist in the
            current video frame.

        Returns
        -------
        detected_pages : dict of (ScreenABC, PageABC or None)
            A map between projections screens that exist in the current video frame and the document
            page detected in a projection screen. If no page was detected in a screen, then the
            screen maps to ``None``.
        """

    def __repr__(self):
        return '<{classname}>'.format(
            classname=self.__class__.__name__,
        )
