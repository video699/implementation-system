# -*- coding: utf-8 -*-

"""This module contains the implementation of a screen detector that uses XML human annotations.

"""

from bisect import bisect
import logging
import os

from dateutil.parser import parse
from lxml import etree

from ..interface import ScreenDetectorABC


LOGGER = logging.getLogger(__name__)
RESOURCES_PATHNAME = os.path.join(os.path.dirname(__file__), 'annotated')
DATASET_PATHNAME = os.path.join(RESOURCES_PATHNAME, 'dataset.xml')


class HumanAnnotationScreenDetector(ScreenDetectorABC):
    """A screen detector that maps video frames to iterables of screens based on human annotations.

    Attributes
    ----------
    institution : str
        An institution identifier.
    room : str
        A room identifier.

    Parameters
    ----------
    institution : str
        An institution identifier.
    room : str
        A room identifier.
    """

    dataset = None

    def __init__(self, institution, room, datetime):
        self._init_dataset()
        assert (institution, room) in self.dataset, \
            'Institution "{}", room "{}" not found in the projection screen dataset'.format(
                institution,
                room
            )
        self.institution = institution
        self.room = room
        self.datetime = datetime  # TODO: Use VideoABC object linked to the FrameABC frame object

    @classmethod
    def _init_dataset(cls):
        if cls.dataset is not None:
            return
        LOGGER.debug('Loading dataset')
        institutions = etree.parse(DATASET_PATHNAME)
        institutions.xinclude()
        cls.dataset = {
            (room.getparent().getparent().attrib['id'], room.attrib['id']): [
                {  # TODO: Use _ScreenPosition object
                    'id': screen.attrib['id'],
                    'from': parse(screen.attrib['from']) if 'from' in screen.attrib else None,
                    'until': parse(screen.attrib['until']) if 'until' in screen.attrib else None,
                    'positions': sorted([
                        (
                            parse(position.attrib['datetime']),
                            (  # TODO: Use BoundingQuadrilinear object
                                (int(position.attrib['x0']), int(position.attrib['y0'])),
                                (int(position.attrib['x1']), int(position.attrib['y1'])),
                                (int(position.attrib['x2']), int(position.attrib['y2'])),
                                (int(position.attrib['x3']), int(position.attrib['y3'])),
                            )
                        ) for position in screen.findall('./positions/position')
                    ]),
                } for screen in room.findall('./screens/screen')
            ] for room in institutions.findall('./institution/rooms/room')
        }

    def detect(self, _):
        for screen in self.dataset[(self.institution, self.room)]:
            if not screen['positions']:
                continue
            current_datetime = self.datetime
            earliest_datetime = screen['positions'][0][0]
            from_datetime = screen['from'] or earliest_datetime
            until_datetime = screen['until']
            if current_datetime < max(from_datetime, earliest_datetime):
                continue
            if until_datetime is not None and current_datetime >= until_datetime:
                continue
            position_index = bisect(
                [position[0] for position in screen['positions']], current_datetime
            ) - 1
            yield (screen['id'], screen['positions'][position_index][1])  # TODO: Use Screen object
