# -*- coding: utf-8 -*-

from io import BytesIO
import os
import unittest

from dateutil.parser import parse as datetime_parse
from lxml import etree
from lxml.etree import xmlfile, XMLSchema

from video699.document.image_file import ImageFileDocument
from video699.event.screen import ScreenEventDetector, ScreenEventDetectorVideo, \
    ScreenAppearedEvent, ScreenChangedContentEvent, ScreenMovedEvent, ScreenDisappearedEvent
from video699.convex_quadrangle import ConvexQuadrangle
from ..document.test_image_file import FIRST_PAGE_IMAGE_PATHNAME, SECOND_PAGE_IMAGE_PATHNAME


VIDEO_FPS = 15
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 576
VIDEO_DATETIME = datetime_parse('2016-10-26T00:00:00+00:00')
FIRST_COORDINATES = ConvexQuadrangle((10, 10), (30, 10), (10, 30), (30, 30))
SECOND_COORDINATES = ConvexQuadrangle((40, 40), (80, 40), (40, 80), (80, 80))
XML_SCHEMA_PATHNAME = os.path.join(os.path.dirname(__file__), 'schema.xsd')


class TestScreenEventDetector(unittest.TestCase):
    """Tests the ability of the ScreenEventDetector class to produce events, and valid XML output.

    """

    def setUp(self):
        image_pathnames = (
            FIRST_PAGE_IMAGE_PATHNAME,
            SECOND_PAGE_IMAGE_PATHNAME,
        )
        self.document = ImageFileDocument(image_pathnames)
        page_iterator = iter(self.document)
        self.first_page = next(page_iterator)
        self.second_page = next(page_iterator)

        self.xml_schema = XMLSchema(file=XML_SCHEMA_PATHNAME)

    def test_empty(self):
        video = ScreenEventDetectorVideo(
            fps=VIDEO_FPS,
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT,
            datetime=VIDEO_DATETIME,
            quadrangles=(),
            pages=(),
        )
        detector = ScreenEventDetector(video)
        screen_events = list(detector)
        self.assertEqual(0, len(screen_events))

        f = BytesIO()
        with xmlfile(f) as xf:
            detector.write_xml(xf)
        f.seek(0)
        xml_document = etree.parse(f)
        self.xml_schema.assertValid(xml_document)

    def test_nonempty(self):
        video = ScreenEventDetectorVideo(
            fps=VIDEO_FPS,
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT,
            datetime=VIDEO_DATETIME,
            quadrangles=(
                FIRST_COORDINATES,
                FIRST_COORDINATES,
                SECOND_COORDINATES,
                SECOND_COORDINATES,
            ),
            pages=(
                self.first_page,
                self.second_page,
                self.second_page,
                self.first_page,
            ),
        )
        detector = ScreenEventDetector(video)
        screen_events = list(detector)
        self.assertEqual(5, len(screen_events))
        screen_event_iterator = iter(screen_events)

        screen_event = next(screen_event_iterator)
        self.assertTrue(isinstance(screen_event, ScreenAppearedEvent))
        self.assertEqual(1, screen_event.frame.number)
        self.assertEqual(FIRST_COORDINATES, screen_event.screen.coordinates)
        self.assertEqual(self.first_page, screen_event.page)

        screen_event = next(screen_event_iterator)
        self.assertTrue(isinstance(screen_event, ScreenChangedContentEvent))
        self.assertEqual(2, screen_event.frame.number)
        self.assertEqual(FIRST_COORDINATES, screen_event.screen.coordinates)
        self.assertEqual(self.second_page, screen_event.page)

        screen_event = next(screen_event_iterator)
        self.assertTrue(isinstance(screen_event, ScreenMovedEvent))
        self.assertEqual(3, screen_event.frame.number)
        self.assertEqual(SECOND_COORDINATES, screen_event.screen.coordinates)

        screen_event = next(screen_event_iterator)
        self.assertTrue(isinstance(screen_event, ScreenChangedContentEvent))
        self.assertEqual(4, screen_event.frame.number)
        self.assertEqual(SECOND_COORDINATES, screen_event.screen.coordinates)
        self.assertEqual(self.first_page, screen_event.page)

        screen_event = next(screen_event_iterator)
        self.assertTrue(isinstance(screen_event, ScreenDisappearedEvent))
        self.assertEqual(5, screen_event.frame.number)
        self.assertEqual(SECOND_COORDINATES, screen_event.screen.coordinates)

        f = BytesIO()
        with xmlfile(f) as xf:
            detector.write_xml(xf)
        f.seek(0)
        xml_document = etree.parse(f)
        self.xml_schema.assertValid(xml_document)


if __name__ == '__main__':
    unittest.main()
