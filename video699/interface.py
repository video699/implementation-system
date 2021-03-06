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

import cv2 as cv

from .common import rescale_and_keep_aspect_ratio, COLOR_RGBA_TRANSPARENT
from .configuration import get_configuration


SCREENABC_CONFIGURATION = get_configuration()['ScreenABC']
SCREENABC_LRU_CACHE_MAXSIZE = SCREENABC_CONFIGURATION.getint('lru_cache_maxsize')
IMAGEABC_CONFIGURATION = get_configuration()['ImageABC']
IMAGEABC_LRU_CACHE_MAXSIZE = IMAGEABC_CONFIGURATION.getint('lru_cache_maxsize')


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
    A video MAY produce *non-consecutive* frames, i.e. frames with non-consecutive but increasing
    frame numbers. This MUST be understood as an indication that no important events took place
    between the two frames.

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


class ImageABC(ABC):
    """An abstract image represented by an OpenCV CV_8UC3 RGBA matrix.

    Attributes
    ----------
    image : array_like
        The image data as an OpenCV CV_8UC3 RGBA matrix, where the alpha channel (A) denotes the
        weight of a pixel. Fully transparent pixels, i.e. pixels with zero alpha, SHOULD be
        completely disregarded in subsequent computation.
    """

    @property
    @abstractmethod
    def image(self):
        pass

    @lru_cache(maxsize=IMAGEABC_LRU_CACHE_MAXSIZE, typed=False)
    def render(self, width=None, height=None):
        """Renders the image at the specified dimensions.

        Parameters
        ----------
        width : int or None, optional
            The width of the image data. When unspecified or ``None``, the implementation MUST pick
            a width at its own discretion.
        height : int or None, optional
            The height of the image data. When unspecified or ``None``, the implementation MUST pick
            a height at its own discretion.

        Returns
        -------
        image : array_like
            The image data of the page as an OpenCV CV_8UC3 RGBA matrix, where the alpha channel (A)
            denotes the weight of a pixel. Fully transparent pixels, i.e. pixels with zero alpha,
            SHOULD be completely disregarded in subsequent computation. Any margins added to the
            image data, e.g. by keeping the aspect ratio of the image, MUST be fully transparent.

        Raises
        ------
        ValueError
            When either the width or the height is zero.
        """

        rgba_image = self.image
        original_height, original_width, _ = rgba_image.shape
        rescaled_width, rescaled_height, top_margin, bottom_margin, left_margin, right_margin = \
            rescale_and_keep_aspect_ratio(original_width, original_height, width, height)
        rescale_interpolation = cv.__dict__[IMAGEABC_CONFIGURATION['rescale_interpolation']]
        rgba_image_rescaled = cv.resize(
            rgba_image,
            (rescaled_width, rescaled_height),
            rescale_interpolation,
        )
        rgba_image_rescaled_with_margins = cv.copyMakeBorder(
            rgba_image_rescaled,
            top_margin,
            bottom_margin,
            left_margin,
            right_margin,
            borderType=cv.BORDER_CONSTANT,
            value=COLOR_RGBA_TRANSPARENT,
        )
        return rgba_image_rescaled_with_margins


@total_ordering
class FrameABC(ImageABC):
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
    duration : timedelta
        The elapsed time since the beginning of the video.
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
    def width(self):
        return self.video.width

    @property
    def height(self):
        return self.video.height

    @property
    def duration(self):
        return timedelta(seconds=(self.number - 1) / self.video.fps)

    @property
    def datetime(self):
        return self.video.datetime + self.duration

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
            The area of the intersection with the other convex quadrangle.
        """
        pass

    @abstractmethod
    def union_area(self, other):
        """The area of the union of two convex quadrangles.

        Parameters
        ----------
        other : ConvexQuadrangleABC
            The other convex quadrangle.

        Returns
        -------
        union_area : scalar
            The area of the union with the other convex quadrangle.
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
    def jaccard_indexes(self, input_quadrangle):
        """Retrieves quadrangles intersecting an input quadrangle, computes Jaccard indexes.

        Parameters
        ----------
        quadrangle : ConvexQuadrangleABC
            An input convex quadrangle.

        Returns
        -------
        jaccard_indexes : dict of (ConvexQuadrangleABC, scalar)
            A map between convex quadrangles in the index that intersect the input convex
            quadrangle, and the Jaccard indexes between the input convex quadrangle and the convex
            quadrangles in the index. All Jaccard indexes MUST be greater than zero.
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
    :class:`MovingConvexQuadrangleABC` objects corresponding to a single conceptual moving
    quadrangle MUST compare equal.

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
            The coordinates of the moving convex quadrangle at the following time frame.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """Produces an iterator of the convex quadrangle movements from the past to the present.

        Returns
        -------
        quadrangles : iterator of ConvexQuadrangleABC
            The coordinates of the moving convex quadrangle from the earliest time frame to the
            current time frame.
        """
        pass

    @abstractmethod
    def __reversed__(self):
        """Produces an iterator of the convex quadrangle movements from the present to the past.

        Returns
        -------
        quadrangles : iterator of ConvexQuadrangleABC
            The coordinates of the moving convex quadrangle from the current time frame to the
            earliest time frame.
        """
        pass

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
    def clear(self):
        """Removes all tracked convex quadrangles.

        """
        pass

    @abstractmethod
    def update(self, quadrangles):
        """Records convex quadrangles that exist in the current time frame.

        Notes
        -----
        The moving convex quadrangles that existed in the previous time frame MUST record the
        coordinates of the moving convex quadrangle at the previous time frame.

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


class ScreenABC(ImageABC):
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
    is_beyond_bounds : bool
        Whether the projection screen extends beyond the bounds of the video frame.
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

    @property
    def is_beyond_bounds(self):
        frame = self.frame
        coordinates = self.coordinates
        return any(
            point[0] < 0 or point[0] >= frame.width or point[1] < 0 or point[1] >= frame.height
            for point in (
                coordinates.top_left,
                coordinates.top_right,
                coordinates.bottom_left,
                coordinates.bottom_right,
            )
        )

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


@total_ordering
class DocumentABC(ABC, Iterable):
    """An abstract text document.

    .. _RFC3987: https://tools.ietf.org/html/rfc3987

    Notes
    -----
    It MUST be possible to repeatedly iterate over all document pages.
    A document MUST contain a page.

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

    def __hash__(self):
        return hash(self.uri)

    def __eq__(self, other):
        if isinstance(other, DocumentABC):
            return self.uri == other.uri
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, DocumentABC):
            return self.uri < other.uri
        return NotImplemented

    def __repr__(self):
        formatted_title = ', "{0}"'.format(self.title) if self.title is not None else ''
        formatted_author = ', "{0}"'.format(self.author) if self.author is not None else ''
        return '<{classname}, {uri}{formatted_author}{formatted_title}>'.format(
            classname=self.__class__.__name__,
            formatted_author=formatted_author,
            formatted_title=formatted_title,
            uri=self.uri,
        )


@total_ordering
class PageABC(ImageABC):
    """An abstract page of a document.

    Attributes
    ----------
    document : DocumentABC
        The document containing the page.
    image : array_like
        The image data of the page as an OpenCV CV_8UC3 RGBA matrix, where the alpha channel (A)
        denotes the weight of a pixel. Fully transparent pixels, i.e. pixels with zero alpha, SHOULD
        be completely disregarded in subsequent computation. Any margins added to the image data,
        e.g. by keeping the aspect ratio of the page, MUST be fully transparent.
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

    def __hash__(self):
        return hash((self.document, self.number))

    def __eq__(self, other):
        if isinstance(other, PageABC):
            return (self.document, self.number) == (other.document, other.number)
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, PageABC):
            return (self.document, self.number) < (other.document, other.number)
        return NotImplemented

    def __repr__(self):
        return '<{classname}, page #{page_number}>'.format(
            classname=self.__class__.__name__,
            page_number=self.number,
        )


class PageDetectorABC(ABC):
    """An abstract detector of document pages in projection screens.

    """

    @abstractmethod
    def detect(self, frame, appeared_screens, existing_screens, disappeared_screens):
        """Detects document pages in projection screens from a current video frame.

        Parameters
        ----------
        frame : FrameABC
            A current video frame.
        appeared_screens : iterator of (ScreenABC, MovingConvexQuadrangleABC)
            Projection screens that did not exist in the previous video frame and exist in the
            current video frame, and their movements.
        existing_screens : iterator of (ScreenABC, MovingConvexQuadrangleABC)
            Projection screens that existed in the previous video frame and exist in the current
            video frame, and their movements.
        disappeared_screens : iterator of (ScreenABC, MovingConvexQuadrangleABC)
            Projection screens that existed in the previous video frame and do not exist in the
            current video frame, and their movements.

        Returns
        -------
        detected_pages : dict of (ScreenABC, PageABC or None)
            A map between projections screens that exist in the current video frame and the document
            page detected in a projection screen. If no page was detected in a screen, then the
            screen maps to ``None``.
        """
        pass

    def __repr__(self):
        return '<{classname}>'.format(
            classname=self.__class__.__name__,
        )
