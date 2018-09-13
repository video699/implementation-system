# -*- coding: utf-8 -*-

"""This module implements a screen detector that uses XML human annotations, and related classes.

"""

from bisect import bisect
from datetime import datetime
from functools import total_ordering
from logging import getLogger
import os

from dateutil.parser import parse as datetime_parse
from lxml import etree
import numpy as np

from ..convex_quadrangle import ConvexQuadrangle
from ..frame.image import ImageFrame
from ..interface import ScreenABC, ScreenDetectorABC, VideoABC


LOGGER = getLogger(__name__)
RESOURCES_PATHNAME = os.path.join(os.path.dirname(__file__), 'annotated')
DATASET_PATHNAME = os.path.join(RESOURCES_PATHNAME, 'dataset.xml')
CAMERA_ANNOTATIONS = None
SCREEN_ANNOTATIONS = None


def _init_dataset():
    """Reads human annotations from an XML dataset, converts them into objects and sorts them.

    """
    global CAMERA_ANNOTATIONS
    global SCREEN_ANNOTATIONS
    LOGGER.debug('Loading dataset {}'.format(DATASET_PATHNAME))
    institutions = etree.parse(DATASET_PATHNAME)
    institutions.xinclude()
    CAMERA_ANNOTATIONS = {
        institution.attrib['id']: {
            room.attrib['id']: {
                camera.attrib['id']: _CameraAnnotations(
                    camera_id=camera.attrib['id'],
                    name=camera.attrib['name'],
                    width=int(camera.attrib['width']),
                    height=int(camera.attrib['height']),
                ) for camera in room.findall('./cameras/camera')
            } for room in institution.findall('./rooms/room')
        } for institution in institutions.findall('./institution')
    }
    SCREEN_ANNOTATIONS = {
        institution.attrib['id']: {
            room.attrib['id']: [
                _ScreenAnnotations(
                    screen_id=screen.attrib['id'],
                    name=screen.attrib['name'],
                    installed_from=datetime_parse(
                        screen.attrib['from']
                    ) if 'from' in screen.attrib else None,
                    installed_until=datetime_parse(
                        screen.attrib['until']
                    ) if 'until' in screen.attrib else None,
                    positions={
                        positions.attrib['camera']: sorted([
                            _ScreenPosition(
                                coordinates=ConvexQuadrangle(
                                    top_left=(
                                        int(position.attrib['x0']),
                                        int(position.attrib['y0']),
                                    ),
                                    top_right=(
                                        int(position.attrib['x1']),
                                        int(position.attrib['y1']),
                                    ),
                                    bottom_left=(
                                        int(position.attrib['x2']),
                                        int(position.attrib['y2']),
                                    ),
                                    bottom_right=(
                                        int(position.attrib['x3']),
                                        int(position.attrib['y3']),
                                    ),
                                ),
                                datetime=datetime_parse(position.attrib['datetime']),
                            ) for position in positions.findall('./position')
                        ]) for positions in screen.findall('./positions')
                    }
                ) for screen in room.findall('./screens/screen')
            ] for room in institution.findall('./rooms/room')
        } for institution in institutions.findall('./institution')
    }


def _assert_key_exists(institution_id, room_id, camera_id):
    """Asserts that annotations exist for a given camcoder in a given room at a given institution.

    Attributes
    ----------
    institution_id : str
        A institution identifier. The identifier is unique in the dataset.
    room_id : str
        A room identifier. The identifier is unique in the institution.
    camera_id : str
        A camcoder identifier. The identifier is unique in the room.

    Raises
    ------
    AssertionError
        If no annotations exist for the given camcoder in the given room at the given institution.
    """

    assert institution_id in SCREEN_ANNOTATIONS, \
        'Institution "{1}" not found in the projection screen dataset'.format(institution_id)
    assert room_id in SCREEN_ANNOTATIONS[institution_id], \
        'Room "{2}" not found in the projection screen dataset for institution "{1}"'.format(
            institution_id,
            room_id,
        )
    assert camera_id in CAMERA_ANNOTATIONS[institution_id][room_id], \
        'Camera "{3}" not found in the projection screen dataset for institution "{1}", ' \
        'room "{2}"'.format(
            institution_id,
            room_id,
            camera_id,
        )


class _CameraAnnotations(object):
    """Human annotations associated with a single camcoder.

    Parameters
    ----------
    camera_id : str
        An identifier of a camcoder in a room. The identifier is unique in the room.
    name : str
        A name of the camcoder.
    width : int
        The width of the captured video in pixels.
    height : int
        The height of the captured video in pixels.

    Attributes
    ----------
    camera_id : str
        An identifier of a camcoder in a room. The identifier is unique in the room.
    name : str
        A name of the camcoder.
    width : int
        The width of the captured video in pixels.
    height : int
        The height of the captured video in pixels.
    """

    def __init__(self, camera_id, name, width, height):
        self.camera_id = camera_id
        self.name = name
        self.width = width
        self.height = height


class _ScreenAnnotations(object):
    """Human annotations associated with a single projection screen.

    Parameters
    ----------
    screen_id : str
        An identifier of a projection screen in a room. The identifier is unique in the room.
    name : str
        A description of the projection screen.
    positions : dict of (str, sequence of _ScreenPosition)
        A map between camcoder identifiers, and human-annotated positions of the projection screen
        in a room at various dates, and times. The positions are sorted by dates, and times in
        ascending order.
    installed_from : datetime or None, optional
        The date, and time that marks the screen's installation. If None, or unspecified, then the
        screen is assumed to be installed at the earliest date and time for which the position of
        the projection screen is known.
    installed_until : datetime or None, optional
        The date, and time that marks the screen's removal. If None, or unspecified, then the screen
        is assumed to be present at any point since the date, and time specified by the (possibly
        inferred) `installed_from` attribute.

    Attributes
    ----------
    screen_id : str
        An identifier of a projection screen in a room. The identifier is unique in the room.
    name : str
        A description of the projection screen.
    positions : dict of (str, sequence of _ScreenPosition)
        A map between camcoder identifiers, and human-annotated positions of the projection screen
        in a room at various dates, and times. The positions are sorted by dates, and times in
        ascending order.
    installed_from : datetime or None
        The date, and time that marks the screen's installation. None if and only if no
        human-annotated positions exist.
    installed_until : datetime or None
        The date, and time that marks the screen's removal. If None, then the screen is assumed to
        be present at any point since the date, and time specified by the (possibly inferred)
        `installed_from` attribute.
    """

    def __init__(self, screen_id, name, positions, installed_from=None, installed_until=None):
        self.screen_id = screen_id
        self.name = name
        if installed_from is None:
            try:
                installed_from = min(
                    position_sequence[0].datetime
                    for position_sequence in positions.values()
                    if position_sequence
                )
            except ValueError:
                pass  # There were no recorded positions
        self.installed_from = installed_from
        self.installed_until = installed_until
        self.positions = positions


@total_ordering
class _ScreenPosition(object):
    """A human-annotated position of a projection screen in a room at a given date, and time.

    Parameters
    ----------
    coordinates : ConvexQuadrangleABC
        A map between frame and screen coordinates.
    datetime : aware datetime
        The human annotation certifies that at this date, and time, the lit projection screen was
        located at the given coordinates.

    Attributes
    ----------
    coordinates : ConvexQuadrangleABC
        A map between frame and screen coordinates.
    datetime : aware datetime
        The human annotation certifies that at this date, and time, the lit projection screen was
        located at the given coordinates.
    """

    def __init__(self, coordinates, datetime):
        self.coordinates = coordinates
        self.datetime = datetime

    def __eq__(self, other):
        if isinstance(other, _ScreenPosition):
            return self.datetime == other.datetime
        elif isinstance(other, datetime):
            return self.datetime == other
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, _ScreenPosition):
            return self.datetime < other.datetime
        elif isinstance(other, datetime):
            return self.datetime < other
        return NotImplemented


class AnnotatedScreenVideo(VideoABC):
    """A video containing a single frame at a given date, and time.

    The width, and the height of the video are extracted from XML human annotations. This is a stub
    class intended for testing purposes.

    Parameters
    ----------
    institution_id : str
        An identifier of the institution where the video was captured.
    room_id : str
        An identifier of the room where the video was captured.
    camera_id : str
        An identifier of the camcoder that was used to capture the video.
    datetime : aware datetime
        The date, and time at which the video was captured.
    fps : int, optional
        The framerate of the video in frames per second.

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
    """

    def __init__(self, institution_id, room_id, camera_id, datetime, fps=1):
        _assert_key_exists(institution_id, room_id, camera_id)
        self._fps = fps
        self._datetime = datetime
        self._camera_annotations = CAMERA_ANNOTATIONS[institution_id][room_id][camera_id]

    @property
    def fps(self):
        return self._fps

    @property
    def width(self):
        return self._camera_annotations.width

    @property
    def height(self):
        return self._camera_annotations.height

    @property
    def datetime(self):
        return self._datetime

    def __iter__(self):
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        frame = ImageFrame(self, 1, image)
        yield frame


class AnnotatedScreen(ScreenABC):
    """A projection screen extracted from XML human annotations.

    Parameters
    ----------
    screen_id : str
        An identifier of a projection screen in a room. The identifier is unique in the room.
    name : str
        A description of the projection screen.
    datetime : aware datetime
        The human annotation certifies that at this date, and time, this projection screen was
        located at the given coordinates.
    frame : FrameABC
        A video frame containing the projection screen.
    coordinates : ConvexQuadrangleABC
        A map between frame and screen coordinates.

    Attributes
    ----------
    screen_id : str
        An identifier of a projection screen in a room. The identifier is unique in the room.
    name : str
        A description of the projection screen.
    datetime : aware datetime
        The human annotation certifies that at this date, and time, this projection screen was
        located at the given coordinates.
    frame : FrameABC
        A video frame containing the projection screen.
    coordinates : ConvexQuadrangleABC
        A map between frame and screen coordinates.
    image : array_like
        The image data of the frame as an OpenCV CV_8UC3 RGBA matrix, where the alpha channel (A)
        is currently unused and all pixels are fully opaque, i.e. they have the maximum alpha of
        255.
    width : int
        The width of the image data.
    height : int
        The height of the image data.
    """

    def __init__(self, screen_id, name, datetime, frame, coordinates):
        self.screen_id = screen_id
        self.name = name
        self.datetime = datetime
        self._frame = frame
        self._coordinates = coordinates

    @property
    def frame(self):
        return self._frame

    @property
    def coordinates(self):
        return self._coordinates


class AnnotatedScreenDetector(ScreenDetectorABC):
    """A screen detector that maps a video frame to screens using XML human annotations.

    Parameters
    ----------
    institution_id : str
        An identifier of the institution where the video was captured.
    room_id : str
        An identifier of the room where the video was captured.
    camera_id : str
        An identifier of the camcoder that was used to capture the video.

    Attributes
    ----------
    institution_id : str
        An identifier of the institution where the video was captured.
    room_id : str
        An identifier of the room where the video was captured.
    camera_id : str
        An identifier of the camcoder that was used to capture the video.
    """

    def __init__(self, institution_id, room_id, camera_id):
        _assert_key_exists(institution_id, room_id, camera_id)
        self.institution_id = institution_id
        self.room_id = room_id
        self.camera_id = camera_id

    def __call__(self, frame):
        """Converts a frame to screens using the closest available human annotations.

        Parameters
        ----------
        frame : FrameABC
            A frame of a video.

        Returns
        -------
        screens : iterable of AnnotatedScreen
            An iterable of detected lit projection screens.
        """
        for screen in SCREEN_ANNOTATIONS[self.institution_id][self.room_id]:
            if (self.camera_id not in screen.positions) or (not screen.positions[self.camera_id]):
                continue
            positions = screen.positions[self.camera_id]
            earliest_position = positions[0]
            if frame.datetime < max(screen.installed_from, earliest_position):
                continue
            if screen.installed_until is not None and frame.datetime >= screen.installed_until:
                continue
            closest_position = positions[bisect(positions, frame.datetime) - 1]
            yield AnnotatedScreen(
                screen_id=screen.screen_id,
                name=screen.name,
                datetime=closest_position.datetime,
                frame=frame,
                coordinates=closest_position.coordinates,
            )


_init_dataset()
