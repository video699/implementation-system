# -*- coding: utf-8 -*-

"""This module implements a screen detector that uses XML human annotations.

"""

from bisect import bisect
from datetime import datetime
from functools import total_ordering
import logging
import os

from dateutil.parser import parse
from lxml import etree

from .coordinate_map.quadrangle import Quadrangle
from .interface import ScreenABC, ScreenDetectorABC


LOGGER = logging.getLogger(__name__)
RESOURCES_PATHNAME = os.path.join(os.path.dirname(__file__), 'annotated')
DATASET_PATHNAME = os.path.join(RESOURCES_PATHNAME, 'dataset.xml')


class _ScreenAnnotations(object):
    """Human annotations associated with a single projection screen.

    Parameters
    ----------
    screen_id : str
        An identifier of a projection screen in a room. The identifier is unique in the room.
    positions : sequence of _ScreenPosition
        Human-annotated positions of the projection screen at various dates, and times.
    installed_from : datetime or None, optional
        The date, and time that marks the screen's installation. If None, or unspecified, then the
        screen is assumed to be installed at the earliest date and time for which the position of
        the projection screen is known.
    installed_until : datetime or None, optional
        The date, and time that marks the screen's removal. If None, or unspecified, then the screen
        is assumed to be present at any point since the date, and time specified by the (possibly
        inferred) `installed_from` attribute.
    name : str or None, optional
        A description of the projection screen. If None or unspecified, no description of the screen
        is known.

    Attributes
    ----------
    screen_id : str
        An identifier of a projection screen in a room. The identifier is unique in the room.
    positions : sequence of _ScreenPosition
        Human-annotated positions of the projection screen at various dates, and times. The
        positions are sorted by dates, and times in ascending order.
    installed_from : datetime or None
        The date, and time that marks the screen's installation. None if and only if no
        human-annotated positions exist.
    installed_until : datetime or None
        The date, and time that marks the screen's removal. If None, then the screen is assumed to
        be present at any point since the date, and time specified by the (possibly inferred)
        `installed_from` attribute.
    name : str or None
        A description of the projection screen. If None, no description of the screen is known.
    """
    def __init__(self, screen_id, positions, installed_from=None, installed_until=None, name=None):
        self.screen_id = screen_id
        self.name = name
        if installed_from is None and positions:
            self.installed_from = positions[0].datetime
        else:
            self.installed_from = installed_from
        self.installed_until = installed_until
        self.positions = sorted(positions)


@total_ordering
class _ScreenPosition(object):
    """A human-annotated position of a projection screen in a room at a given date, and time.

    Parameters
    ----------
    coordinates : CoordinateMapABC
        A map between frame and screen coordinates.
    datetime : aware datetime
        The human annotation certifies that at this date, and time, the lit projection screen was
        located at the given coordinates.

    Attributes
    ----------
    coordinates : CoordinateMapABC
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
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, _ScreenPosition):
            return self.datetime < other.datetime
        elif isinstance(other, datetime):
            return self.datetime < other
        else:
            return NotImplemented


class AnnotatedScreen(ScreenABC):
    """A projection screen extracted from XML human annotations.

    Parameters
    ----------
    screen_id : str
        An identifier of a projection screen in a room. The identifier is unique in the room.
    datetime : aware datetime
        The human annotation certifies that at this date, and time, this projection screen was
        located at the given coordinates.
    frame : FrameABC
        A frame containing the projection screen.
    coordinates : CoordinateMapABC
        A map between frame and screen coordinates.
    name : str or None, optional
        A description of the projection screen. If None or unspecified, no description of the screen
        is known.

    Attributes
    ----------
    screen_id : str
        An identifier of a projection screen in a room. The identifier is unique in the room.
    datetime : aware datetime
        The human annotation certifies that at this date, and time, this projection screen was
        located at the given coordinates.
    frame : FrameABC
        A frame containing the projection screen.
    coordinates : CoordinateMapABC
        A map between frame and screen coordinates.
    name : str or None, optional
        A description of the projection screen. If None or unspecified, no description of the screen
        is known.
    """

    @property
    def frame(self):
        return self._frame

    @property
    def coordinates(self):
        return self._coordinates

    def __init__(self, screen_id, datetime, frame, coordinates, name):
        self.screen_id = screen_id
        self.datetime = datetime
        self._frame = frame
        self._coordinates = coordinates
        self.name = name


class AnnotatedScreenDetector(ScreenDetectorABC):
    """A screen detector that maps video frames to iterables of screens based on human annotations.

    Attributes
    ----------
    institution_id : str
        A institution identifier. The identifier is globally unique.
    room_id : str
        A room identifier. The identifier is unique in the institution.

    Parameters
    ----------
    institution_id : str
        A institution identifier. The identifier is globally unique.
    room_id : str
        A room identifier. The identifier is unique in the institution.
    """

    _dataset = None

    def __init__(self, institution_id, room_id):
        self._init_dataset()
        assert institution_id in self._dataset, \
            'Institution "{}" not found in the projection screen dataset'.format(institution_id)
        assert room_id in self._dataset[institution_id], \
            'Room "{}" not found in the projection screen dataset'.format(room_id)
        self.institution_id = institution_id
        self.room_id = room_id

    @classmethod
    def _init_dataset(cls):
        """Reads human annotations from an XML database, converts them into objects and sorts them.

        """
        if cls._dataset is not None:
            return
        LOGGER.debug('Loading dataset from {}'.format(DATASET_PATHNAME))
        institutions = etree.parse(DATASET_PATHNAME)
        institutions.xinclude()
        cls._dataset = {
            institution.attrib['id']: {
                room.attrib['id']: [
                    _ScreenAnnotations(
                        screen_id=screen.attrib['id'],
                        name=screen.attrib['name'],
                        installed_from=parse(
                            screen.attrib['from']
                        ) if 'from' in screen.attrib else None,
                        installed_until=parse(
                            screen.attrib['until']
                        ) if 'until' in screen.attrib else None,
                        positions=[
                            _ScreenPosition(
                                coordinates=Quadrangle(
                                    top_left=(
                                        int(position.attrib['x0']),
                                        int(position.attrib['y0']),
                                    ),
                                    top_right=(
                                        int(position.attrib['x1']),
                                        int(position.attrib['y1']),
                                    ),
                                    btm_left=(
                                        int(position.attrib['x2']),
                                        int(position.attrib['y2']),
                                    ),
                                    btm_right=(
                                        int(position.attrib['x3']),
                                        int(position.attrib['y3']),
                                    ),
                                ),
                                datetime=parse(position.attrib['datetime']),
                            ) for position in screen.findall('./positions/position')
                        ],
                    ) for screen in room.findall('./screens/screen')
                ] for room in institution.findall('./rooms/room')
            } for institution in institutions.findall('./institution')
        }

    def __call__(self, frame):
        """Converts a frame to an iterable of screens using the closest available human annotations.

        Parameters
        ----------
        frame : FrameABC
            A frame of a video.

        Returns
        -------
        screens : iterable of AnnotatedScreen
            An iterable of detected lit projection screens.
        """
        for screen in self._dataset[self.institution_id][self.room_id]:
            if not screen.positions:
                continue
            earliest_position = screen.positions[0]
            if frame.datetime < max(screen.installed_from, earliest_position):
                continue
            if screen.installed_until is not None and frame.datetime >= screen.installed_until:
                continue
            closest_position = screen.positions[
                bisect(screen.positions, frame.datetime) - 1
            ]
            yield AnnotatedScreen(
                screen_id=screen.screen_id,
                datetime=closest_position.datetime,
                frame=frame,
                coordinates=closest_position.coordinates,
                name=screen.name,
            )
