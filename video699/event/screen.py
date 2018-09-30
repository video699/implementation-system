# -*- coding: utf-8 -*-

"""This module defines events describing the detection of the appearance and the disappearance of
projection screens showing document pages, and the detection of changes in the screen coordinates
and content. Related classes are also defined.

"""

from abc import abstractmethod
from itertools import chain
from logging import getLogger
from lxml.etree import Element

import numpy as np

from ..common import timedelta_as_xsd_duration
from ..interface import EventDetectorABC, VideoABC, ScreenABC, ScreenDetectorABC, PageDetectorABC
from .frame import FrameEventABC
from ..frame.image import ImageFrame


LOGGER = getLogger(__name__)


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
        page = self.page
        document = page.document
        xsd_duration = timedelta_as_xsd_duration(frame.duration)

        xml_element = Element('screen-appeared-event')
        xml_element.attrib['screen-id'] = self.screen_id
        xml_element.attrib['frame-number'] = str(frame.number)
        xml_element.attrib['frame-duration'] = xsd_duration
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
        page = self.page
        document = page.document
        xsd_duration = timedelta_as_xsd_duration(frame.duration)

        xml_element = Element('screen-changed-content-event')
        xml_element.attrib['screen-id'] = self.screen_id
        xml_element.attrib['frame-number'] = str(frame.number)
        xml_element.attrib['frame-duration'] = xsd_duration
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
        xsd_duration = timedelta_as_xsd_duration(frame.duration)

        xml_element = Element('screen-moved-event')
        xml_element.attrib['screen-id'] = self.screen_id
        xml_element.attrib['frame-number'] = str(frame.number)
        xml_element.attrib['frame-duration'] = xsd_duration

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
    frame : FrameABC
        A frame in which the event takes place.
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

    def __init__(self, frame, screen, screen_id):
        self._frame = frame
        self._screen = screen
        self._screen_id = screen_id

    @property
    def frame(self):
        return self._frame

    @property
    def screen(self):
        return self._screen

    @property
    def screen_id(self):
        return self._screen_id

    def write_xml(self, xf):
        frame = self.frame
        xsd_duration = timedelta_as_xsd_duration(frame.duration)

        xml_element = Element('screen-disappeared-event')
        xml_element.attrib['screen-id'] = self.screen_id
        xml_element.attrib['frame-number'] = str(frame.number)
        xml_element.attrib['frame-duration'] = xsd_duration

        xf.write(xml_element)


class ScreenEventDetectorScreen(ScreenABC):
    """A projection screen shown in a frame of a :class:`ScreenEventDetectorVideo` video.

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
    """A video whose frames show a projection screen that changes coordinates and content.

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
    duration : timedelta
        The elapsed time since the beginning of the video.
    datetime : aware datetime
        The date, and time at which the video was captured.
    quadrangles : sequence of ConvexQuadrangleABC
        The projection screen coordinates in the consecutive frames of the video.
    pages : sequence of PageABC
        The document pages shown in the projection screen in the consecutive frames of the video.

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
    screens : dict of (FrameABC, ScreenABC)
        A map between video frames, and the projection screens shown in the frames.
    pages : dict of (ScreenABC, PageABC)
        A map between projection screens, and the document pages shown in the screens.
    """

    _num_videos = 0

    def __init__(self, fps, width, height, datetime, quadrangles, pages):
        self._fps = fps
        self._width = width
        self._height = height
        self._datetime = datetime

        assert len(quadrangles) == len(pages)
        num_frames = len(quadrangles) + 1 if quadrangles else 0
        image = np.zeros((height, width, 4), dtype=np.uint8)
        self._frames = [
            ImageFrame(self, frame_number, image)
            for frame_number in range(1, num_frames + 1)
        ]
        self.screens = {
            frame: ScreenEventDetectorScreen(frame, coordinates)
            for frame, coordinates in zip(self, quadrangles)
        }
        self.pages = {
            self.screens[frame]: page
            for frame, page in zip(self, pages)
        }

        self._uri = 'https://github.com/video699/implementation-system/blob/master/video699/' \
            'event/screen.py#ScreenEventDetectorVideo:{}'.format(self._num_videos + 1)
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


class ScreenEventDetectorScreenDetector(ScreenDetectorABC):
    """A detector of screens in the frames of a :class:`ScreenEventDetectorVideo` video.

    Notes
    -----
    This is a stub class intended for testing.

    """

    def detect(self, frame):
        video = frame.video
        if isinstance(video, ScreenEventDetectorVideo) and frame in video.screens:
            return (video.screens[frame],)
        return ()


class ScreenEventDetectorPageDetector(PageDetectorABC):
    """A detector of pages in the screens shown in :class:`ScreenEventDetectorVideo` video frames.

    Notes
    -----
    This is a stub class intended for testing.

    """

    def detect(self, frame, appeared_screens, existing_screens, disappeared_screens):
        detected_pages = {}

        for screen, _ in chain(appeared_screens, existing_screens):
            frame = screen.frame
            video = frame.video
            if isinstance(video, ScreenEventDetectorVideo) and screen in video.pages:
                detected_pages[screen] = video.pages[screen]
            else:
                detected_pages[screen] = None

        return detected_pages


class ScreenEventDetector(EventDetectorABC):
    r"""An event detector that detects screen events in a video.

    In each frame of a video, screens are detected using a provided screen detector. The detected
    screens are tracked over time using a provided convex quadrangle tracker. New screens that
    *match* a document page as determined by a provided page tracker, or screens that have already
    been known to the quadrangle tracker but that previously did not match a document page are
    reported in a :class:`ScreenAppearedEvent` event. A change of coordinates of a known screen is
    reported in a :class:`ScreenMovedEvent` event and a change of the matching page of a known
    screen is reported in a :class:`ScreenChangedContentEvent` event. Known screens that no longer
    match a page are reported in a :class:`ScreenDisappearedEvent` event.

    Notes
    -----
    It is not possible to repeatedly iterate over all events detected in the video.
    Iterating over frames in the video prevents iterating over all events detected in the video.
    For screens that do not disappear by the last frame of the video, a
    :class:`ScreenDisappearedEvent` event will not be produced.

    Parameters
    ----------
    video : VideoABC
        The video in which the events are detected.
    quadrangle_tracker : ConvexQuadrangleTrackerABC
        The provided convex quadrangle tracker. If non-empty, the tracker will be cleared.
    screen_detector : ScreenDetectorABC
        The provided screen detector that will be used to detect lit projection screens in video
        frames.
    page_detector : PageDetectorABC
        The provided page detector that will be used to determine whether a screen shows a document
        page.

    Attributes
    ----------
    video : VideoABC
        The video in which the events are detected.
    """

    def __init__(self, video, quadrangle_tracker, screen_detector, page_detector):
        self._video = video
        if quadrangle_tracker:
            quadrangle_tracker.clear()
        self._quadrangle_tracker = quadrangle_tracker
        self._screen_detector = screen_detector
        self._page_detector = page_detector

    @property
    def video(self):
        return self._video

    def __iter__(self):
        quadrangle_tracker = self._quadrangle_tracker
        screen_detector = self._screen_detector
        page_detector = self._page_detector

        num_screens = 0
        screen_ids = {}
        matched_pages = {}
        matched_quadrangles = matched_pages.keys()
        detected_screens = None
        previous_detected_screens = None

        for frame in self.video:
            previous_detected_screens = detected_screens
            detected_screens = {
                screen.coordinates: screen
                for screen in screen_detector.detect(frame)
            }
            detected_quadrangles = detected_screens.keys()
            appeared_quadrangles, existing_quadrangles, disappeared_quadrangles = \
                quadrangle_tracker.update(detected_quadrangles)

            for moving_quadrangle in sorted(disappeared_quadrangles):
                if moving_quadrangle in matched_quadrangles:
                    quadrangle = moving_quadrangle.current_quadrangle
                    screen = previous_detected_screens[quadrangle]
                    screen_id = screen_ids[moving_quadrangle]
                    previous_page = matched_pages[moving_quadrangle]
                    del screen_ids[moving_quadrangle]
                    del matched_pages[moving_quadrangle]
                    LOGGER.debug('{} disappeared matching {}'.format(screen, previous_page))
                    yield ScreenDisappearedEvent(frame, screen, screen_id)
                else:
                    LOGGER.debug('{} disappeared with no matching page'.format(screen))

            moving_quadrangles = {
                moving_quadrangle.current_quadrangle: moving_quadrangle
                for moving_quadrangle in chain(
                    appeared_quadrangles,
                    existing_quadrangles,
                )
            }

            def moving_quadrangle_to_screen(moving_quadrangle):
                return (detected_screens[moving_quadrangle.current_quadrangle], moving_quadrangle)

            def moving_quadrangle_to_previous_screen(moving_quadrangle):
                return (
                    previous_detected_screens[moving_quadrangle.current_quadrangle],
                    moving_quadrangle,
                )

            pages = page_detector.detect(
                frame,
                map(moving_quadrangle_to_screen, appeared_quadrangles),
                map(moving_quadrangle_to_screen, existing_quadrangles),
                map(moving_quadrangle_to_previous_screen, disappeared_quadrangles),
            )

            for screen, page in pages.items():
                moving_quadrangle = moving_quadrangles[screen.coordinates]
                quadrangle_iter = reversed(moving_quadrangle)
                current_quadrangle = next(quadrangle_iter)
                if moving_quadrangle in existing_quadrangles:
                    previous_quadrangle = next(quadrangle_iter)

                if page:
                    if moving_quadrangle in appeared_quadrangles:
                        LOGGER.debug('{} appeared and matches {}'.format(screen, page))
                        screen_id = 'screen-{}'.format(num_screens + 1)
                        num_screens += 1
                        screen_ids[moving_quadrangle] = screen_id
                        yield ScreenAppearedEvent(screen, screen_id, page)
                    elif moving_quadrangle in existing_quadrangles:
                        if moving_quadrangle in matched_quadrangles:
                            screen_id = screen_ids[moving_quadrangle]
                            previous_page = matched_pages[moving_quadrangle]
                            if previous_page != page:
                                LOGGER.debug(
                                    '{} changed content from {} to {}'.format(
                                        screen,
                                        previous_page,
                                        page,
                                    )
                                )
                                yield ScreenChangedContentEvent(screen, screen_id, page)
                            if current_quadrangle != previous_quadrangle:
                                LOGGER.debug(
                                    '{} moved from {} to {}'.format(
                                        screen,
                                        previous_quadrangle,
                                        current_quadrangle,
                                    )
                                )
                                yield ScreenMovedEvent(screen, screen_id)
                        else:
                            LOGGER.debug(
                                '{} started matching {}'.format(
                                    screen,
                                    page,
                                )
                            )
                            screen_id = 'screen-{}'.format(num_screens)
                            num_screens += 1
                            screen_ids[moving_quadrangle] = screen_id
                            yield ScreenAppearedEvent(screen, screen_id, page)
                    matched_pages[moving_quadrangle] = page
                else:
                    if moving_quadrangle in appeared_quadrangles:
                        LOGGER.debug('{} appeared with no matching page'.format(screen))
                    elif moving_quadrangle in existing_quadrangles:
                        if moving_quadrangle in matched_quadrangles:
                            screen_id = screen_ids[moving_quadrangle]
                            previous_page = matched_pages[moving_quadrangle]
                            LOGGER.debug('{} no longer matches {}'.format(screen, previous_page))
                            del screen_ids[moving_quadrangle]
                            del matched_pages[moving_quadrangle]
                            yield ScreenDisappearedEvent(frame, screen, screen_id)
