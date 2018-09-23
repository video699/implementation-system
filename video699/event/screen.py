# -*- coding: utf-8 -*-

"""This module defines events describing the detection of the appearance and the disappearance of
projection screens showing document pages, and the detection of changes in the screen coordinates
and content. Related classes are also defined.

"""

from abc import abstractmethod
from itertools import chain, repeat
import numpy as np
from lxml.etree import Element

from ..interface import EventDetectorABC, VideoABC, ScreenABC
from .frame import FrameEventABC
from ..frame.image import ImageFrame
from ..common import timedelta_as_xsd_duration

SCREEN_EVENT_DETECTOR_SCREEN_ID = 'screen'


class ScreenEventABC(FrameEventABC):
    """An event related to a projection screen shown in a video frame.

    Notes
    -----
    Other screen events MUST NOT take place in the same frame as a :class:`ScreenAppearedEvent`
    event.

    Attributes
    ----------
    frame : FrameABC
        A frame in which the event takes place.
    screen : ScreenABC
        A projection screen shown in a video frame.
    screen_id : str
        A screen identifier.
    """

    @property
    def frame(self):
        return self.screen.frame

    @property
    @abstractmethod
    def screen(self):
        pass

    @property
    @abstractmethod
    def screen_id(self):
        pass

    def __repr__(self):
        return '<{classname}, {screen_id}, screen {screen}>'.format(
            classname=self.__class__.__name__,
            screen=self.screen,
            screen_id=self.screen_id,
            quadrangle_id=self.quadrangle_id,
        )


class ScreenAppearedEvent(ScreenEventABC):
    """The appearance of a projection screen containing a document page.

    Parameters
    ----------
    screen : ScreenABC
        A detected projection screen containing a document page.
    screen_id : string
        A screen identifier unique among the :class:`ScreenAppearedEvent` events produced by an
        event detector.
    page : PageABC
        The document page.

    Attributes
    ----------
    frame : FrameABC
        A frame in which the event takes place.
    screen : ScreenABC
        A detected projection screen containing a document page.
    screen_id : str
        A screen identifier unique among the :class:`ScreenAppearedEvent` events produced by an
        event detector.
    page : PageABC
        The document page.
    """

    def __init__(self, screen, screen_id, page):
        self._screen = screen
        self._screen_id = screen_id
        self.page = page

    @property
    def screen(self):
        return self._screen

    @property
    def screen_id(self):
        return self._screen_id

    def write_xml(self, xf):
        screen = self.screen
        frame = self.frame
        video = frame.video
        page = self.page
        document = page.document
        xsd_duration = timedelta_as_xsd_duration(frame.datetime - video.datetime)

        xml_element = Element('screen-appeared-event')
        xml_element.attrib['screen-id'] = self.screen_id
        xml_element.attrib['frame-number'] = str(frame.number)
        xml_element.attrib['frame-time'] = xsd_duration
        xml_element.attrib['document-uri'] = document.uri
        xml_element.attrib['page-number'] = str(page.number)

        x0, y0 = screen.coordinates.top_left
        xml_element.attrib['x0'] = str(x0)
        xml_element.attrib['y0'] = str(y0)
        x1, y1 = screen.coordinates.top_right
        xml_element.attrib['x1'] = str(x1)
        xml_element.attrib['y1'] = str(y1)
        x2, y2 = screen.coordinates.bottom_left
        xml_element.attrib['x2'] = str(x2)
        xml_element.attrib['y2'] = str(y2)
        x3, y3 = screen.coordinates.bottom_right
        xml_element.attrib['x3'] = str(x3)
        xml_element.attrib['y3'] = str(y3)

        xf.write(xml_element)


class ScreenChangedContentEvent(ScreenEventABC):
    """A change of the document page shown in a projection screen.

    Parameters
    ----------
    screen : ScreenABC
        A projection screen, which now shows a different document page.
    screen_id : string
        A screen identifier that MUST have appeared in an earlier :class:`ScreenAppearedEvent` event
        produced by an event detector.
    page : PageABC
        The different document page.

    Attributes
    ----------
    frame : FrameABC
        A frame in which the event takes place.
    screen : ScreenABC
        A projection screen, which now shows a different document page.
    screen_id : string
        A screen identifier that MUST have appeared in an earlier :class:`ScreenAppearedEvent` event
        produced by an event detector,
    page : PageABC
        The different document page.
    """

    def __init__(self, screen, screen_id, page):
        self._screen = screen
        self._screen_id = screen_id
        self.page = page

    @property
    def screen(self):
        return self._screen

    @property
    def screen_id(self):
        return self._screen_id

    def write_xml(self, xf):
        frame = self.frame
        video = frame.video
        page = self.page
        document = page.document
        xsd_duration = timedelta_as_xsd_duration(frame.datetime - video.datetime)

        xml_element = Element('screen-changed-content-event')
        xml_element.attrib['screen-id'] = self.screen_id
        xml_element.attrib['frame-number'] = str(frame.number)
        xml_element.attrib['frame-time'] = xsd_duration
        xml_element.attrib['document-uri'] = document.uri
        xml_element.attrib['page-number'] = str(page.number)

        xf.write(xml_element)


class ScreenMovedEvent(ScreenEventABC):
    """A change in the coordinates of a projection screen.

    Parameters
    ----------
    screen : ScreenABC
        A projection screen, which now appears at different coordinates.
    screen_id : string
        A screen identifier that MUST have appeared in an earlier :class:`ScreenAppearedEvent` event
        produced by an event detector.

    Attributes
    ----------
    frame : FrameABC
        A frame in which the event takes place.
    screen : ScreenABC
        A projection screen, which now appears at different coordinates.
    screen_id : string
        A screen identifier that MUST have appeared in an earlier :class:`ScreenAppearedEvent` event
        produced by an event detector.
    """

    def __init__(self, screen, screen_id):
        self._screen = screen
        self._screen_id = screen_id

    @property
    def screen(self):
        return self._screen

    @property
    def screen_id(self):
        return self._screen_id

    def write_xml(self, xf):
        screen = self.screen
        frame = self.frame
        video = frame.video
        xsd_duration = timedelta_as_xsd_duration(frame.datetime - video.datetime)

        xml_element = Element('screen-moved-event')
        xml_element.attrib['screen-id'] = self.screen_id
        xml_element.attrib['frame-number'] = str(frame.number)
        xml_element.attrib['frame-time'] = xsd_duration

        x0, y0 = screen.coordinates.top_left
        xml_element.attrib['x0'] = str(x0)
        xml_element.attrib['y0'] = str(y0)
        x1, y1 = screen.coordinates.top_right
        xml_element.attrib['x1'] = str(x1)
        xml_element.attrib['y1'] = str(y1)
        x2, y2 = screen.coordinates.bottom_left
        xml_element.attrib['x2'] = str(x2)
        xml_element.attrib['y2'] = str(y2)
        x3, y3 = screen.coordinates.bottom_right
        xml_element.attrib['x3'] = str(x3)
        xml_element.attrib['y3'] = str(y3)

        xf.write(xml_element)


class ScreenDisappearedEvent(ScreenEventABC):
    """The disappearance of a projection screen.

    Notes
    -----
    Other screen events MUST NOT take place in the same frame as a :class:`ScreenDisappearedEvent`
    event.  For every :class:`ScreenAppearedEvent` event, a :class:`ScreenDisappearedEvent` event
    with the same screen identifier MAY be produced by an event detector.

    Parameters
    ----------
    screen : ScreenABC
        A disappeared projection screen.
    screen_id : str
        A screen identifier unique among the :class:`ScreenDisappearedEvent` events produced by an
        event detector. The identifier MUST have appeared in an earlier :class:`ScreenAppearedEvent`
        produced by an event detector.

    Attributes
    ----------
    frame : FrameABC
        A frame in which the event takes place.
    screen : ScreenABC
        A disappeared projection screen.
    screen_id : string
        A screen identifier unique among the :class:`ScreenDisappearedEvent` events produced by an
        event detector. The identifier MUST have appeared in an earlier :class:`ScreenAppearedEvent`
        produced by an event detector.
    """

    def __init__(self, screen, screen_id):
        self._screen = screen
        self._screen_id = screen_id

    @property
    def screen(self):
        return self._screen

    @property
    def screen_id(self):
        return self._screen_id

    def write_xml(self, xf):
        frame = self.frame
        video = frame.video
        xsd_duration = timedelta_as_xsd_duration(frame.datetime - video.datetime)

        xml_element = Element('screen-disappeared-event')
        xml_element.attrib['screen-id'] = self.screen_id
        xml_element.attrib['frame-number'] = str(frame.number)
        xml_element.attrib['frame-time'] = xsd_duration

        xf.write(xml_element)


class ScreenEventDetectorScreen(ScreenABC):
    """A projection screen shown in a :class:`ScreenEventDetectorVideo` video.

    Notes
    -----
    This is a stub class intended for testing.

    Parameters
    ----------
    frame : FrameABC
        A video frame containing the projection screen.
    coordinates : ConvexQuadrangleABC
        A map between frame and screen coordinates.

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

    def __init__(self, frame, coordinates):
        self._frame = frame
        self._coordinates = coordinates

    @property
    def frame(self):
        return self._frame

    @property
    def coordinates(self):
        return self._coordinates


class ScreenEventDetectorVideo(VideoABC):
    """A video containing frames showing a projection screen changing coordinates and content.

    Notes
    -----
    It is possible to repeatedly iterate over all video frames.
    This is a stub class intended for testing.

    .. _RFC3987: https://tools.ietf.org/html/rfc3987

    Parameters
    ----------
    fps : scalar
        The framerate of the video in frames per second.
    width : int
        The width of the video.
    height : int
        The height of the video.
    datetime : aware datetime
        The date, and time at which the video was captured.
    quadrangles : iterable of ConvexQuadrangleABC
        Maps between frame and screen coordinates in consecutive frames of the video.
    pages : iterable of PageABC
        The document pages shown in the projection screen in consecutive frames of the video.

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
    screen_events : sequence of ScreenEventABC
        Screen events that take place in consecutive frames of the video.
    """

    _num_videos = 0

    def __init__(self, fps, width, height, datetime, quadrangles, pages):
        self._fps = fps
        self._width = width
        self._height = height
        self._datetime = datetime

        image = np.zeros((height, width, 4), dtype=np.uint8)
        quadrangles_list = list(quadrangles)
        pages_list = list(pages)
        assert len(quadrangles_list) == len(pages_list)
        num_frames = len(quadrangles_list) + 1 if quadrangles else 0
        self._frames = [
            ImageFrame(self, frame_number, image)
            for frame_number in range(1, num_frames + 1)
        ]
        if num_frames:
            last_coordinates = quadrangles_list[-1]
            padded_quadrangles = chain(quadrangles_list, repeat(last_coordinates))
            screens = (
                ScreenEventDetectorScreen(frame, coordinates)
                for frame, coordinates in zip(self, padded_quadrangles)
            )
            last_page = pages_list[-1]
            padded_pages = chain(pages_list, repeat(last_page))
        else:
            screens = ()
            padded_pages = ()
        screen_id = SCREEN_EVENT_DETECTOR_SCREEN_ID
        previous_screen = None
        previous_page = None
        self.screen_events = []
        for screen, page in zip(screens, padded_pages):
            frame_number = screen.frame.number
            if frame_number == 1:
                screen_event = ScreenAppearedEvent(screen, screen_id, page)
                self.screen_events.append(screen_event)
            elif frame_number == num_frames:
                screen_event = ScreenDisappearedEvent(screen, screen_id)
                self.screen_events.append(screen_event)
            else:
                if page != previous_page:
                    screen_event = ScreenChangedContentEvent(screen, screen_id, page)
                    self.screen_events.append(screen_event)
                if screen.coordinates != previous_screen.coordinates:
                    screen_event = ScreenMovedEvent(screen, screen_id)
                    self.screen_events.append(screen_event)
            previous_screen = screen
            previous_page = page

        self._uri = 'https://github.com/video699/implementation-system/blob/master/video699/' \
            'event/screen.py#ScreenEventDetectorVideo:{}'.format(self._num_videos)
        self._num_videos += 1

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
        return iter(self._frames)


class ScreenEventDetector(EventDetectorABC):
    """A detector of screen events in a :class:`ScreenEventDetectorVideo` video.

    Notes
    -----
    It is possible to repeatedly iterate over all events detected in a video.
    This is a stub class intended for testing.

    Parameters
    ----------
    video : ScreenEventDetectorVideo
        The video in which the events are detected.

    Attributes
    ----------
    video : ScreenEventDetectorVideo
        The video in which the events are detected.
    """

    def __init__(self, video):
        self._video = video

    @property
    def video(self):
        return self._video

    def __iter__(self):
        if not isinstance(self.video, ScreenEventDetectorVideo):
            return iter(())
        return iter(self.video.screen_events)
