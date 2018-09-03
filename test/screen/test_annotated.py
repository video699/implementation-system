# -*- coding: utf-8 -*-

import unittest

from dateutil.parser import parse
from video699.screen.annotated import AnnotatedScreenDetector, AnnotatedVideo


INSTITUTION_ID = 'example'
ROOM_ID = '123'
CAMERA_ID = 'xm2'
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 576


class TestAnnotatedVideo(unittest.TestCase):
    """Tests the ability of the AnnotatedVideo class to detect its dimensions and to produce frames.

    """
    def test_video_dimensions(self):
        datetime = parse('2018-01-01T00:00:00+00:00')
        video = AnnotatedVideo(INSTITUTION_ID, ROOM_ID, CAMERA_ID, datetime)
        self.assertEqual(VIDEO_WIDTH, video.width)
        self.assertEqual(VIDEO_HEIGHT, video.height)

    def test_produces_single_frame(self):
        datetime = parse('2018-01-01T00:00:00+00:00')
        video = AnnotatedVideo(INSTITUTION_ID, ROOM_ID, CAMERA_ID, datetime)
        frame_iterator = video.__iter__()
        next(frame_iterator)
        with self.assertRaises(StopIteration):
            next(frame_iterator)


class TestAnnotatedScreenDetector(unittest.TestCase):
    """Tests the ability of the AnnotatedScreenDetector class to read the example XML dataset.

    """
    def setUp(self):
        self.screen_detector = AnnotatedScreenDetector(INSTITUTION_ID, ROOM_ID, CAMERA_ID)

    def test_no_screens_before_earliest_datetime(self):
        datetime = parse('2017-12-31T23:59:59+00:00')
        video = AnnotatedVideo(INSTITUTION_ID, ROOM_ID, CAMERA_ID, datetime)
        frame = next(video.__iter__())
        screens = set(self.screen_detector(frame))
        self.assertEqual(screens, set())

    def test_screens_at_earliest_datetime(self):
        datetime = parse('2018-01-01T00:00:00+00:00')
        video = AnnotatedVideo(INSTITUTION_ID, ROOM_ID, CAMERA_ID, datetime)
        frame = next(video.__iter__())
        self.assertEqual(
            set([
                (screen.screen_id, screen.datetime)
                for screen in self.screen_detector(frame)
            ]),
            set([
                (
                    'no_from_no_until',
                    parse('2018-01-01T00:00:00+00:00')
                ),
                (
                    'early_from_no_until',
                    parse('2018-01-01T00:00:00+00:00')
                ),
                (
                    'equal_from_no_until',
                    parse('2018-01-01T00:00:00+00:00')
                ),
                (
                    'no_from_early_until',
                    parse('2018-01-01T00:00:00+00:00')
                ),
                (
                    'no_from_equal_until',
                    parse('2018-01-01T00:00:00+00:00')
                ),
                (
                    'no_from_late_until',
                    parse('2018-01-01T00:00:00+00:00')
                ),
            ]),
        )

    def test_screens_after_earliest_datetime(self):
        datetime = parse('2018-01-01T00:00:01+00:00')
        video = AnnotatedVideo(INSTITUTION_ID, ROOM_ID, CAMERA_ID, datetime)
        frame = next(video.__iter__())
        self.assertEqual(
            set([
                (screen.screen_id, screen.datetime)
                for screen in self.screen_detector(frame)
            ]),
            set([
                (
                    'no_from_no_until',
                    parse('2018-01-01T00:00:00+00:00')
                ),
                (
                    'early_from_no_until',
                    parse('2018-01-01T00:00:00+00:00')
                ),
                (
                    'equal_from_no_until',
                    parse('2018-01-01T00:00:00+00:00')
                ),
                (
                    'late_from_no_until',
                    parse('2018-01-01T00:00:00+00:00')
                ),
                (
                    'no_from_early_until',
                    parse('2018-01-01T00:00:00+00:00')
                ),
                (
                    'no_from_equal_until',
                    parse('2018-01-01T00:00:00+00:00')
                ),
                (
                    'no_from_late_until',
                    parse('2018-01-01T00:00:00+00:00')
                ),
            ]),
        )

    def test_screens_before_latest_datetime(self):
        datetime = parse('2018-02-28T23:59:59+00:00')
        video = AnnotatedVideo(INSTITUTION_ID, ROOM_ID, CAMERA_ID, datetime)
        frame = next(video.__iter__())
        self.assertEqual(
            set([
                (screen.screen_id, screen.datetime)
                for screen in self.screen_detector(frame)
            ]),
            set([
                (
                    'no_from_no_until',
                    parse('2018-02-01T00:00:00+00:00')
                ),
                (
                    'early_from_no_until',
                    parse('2018-02-01T00:00:00+00:00')
                ),
                (
                    'equal_from_no_until',
                    parse('2018-02-01T00:00:00+00:00')
                ),
                (
                    'late_from_no_until',
                    parse('2018-02-01T00:00:00+00:00')
                ),
                (
                    'no_from_equal_until',
                    parse('2018-02-01T00:00:00+00:00')
                ),
                (
                    'no_from_late_until',
                    parse('2018-02-01T00:00:00+00:00')
                ),
            ]),
        )

    def test_screens_at_latest_datetime(self):
        datetime = parse('2018-03-01T00:00:00+00:00')
        video = AnnotatedVideo(INSTITUTION_ID, ROOM_ID, CAMERA_ID, datetime)
        frame = next(video.__iter__())
        self.assertEqual(
            set([
                (screen.screen_id, screen.datetime)
                for screen in self.screen_detector(frame)
            ]),
            set([
                (
                    'no_from_no_until',
                    parse('2018-03-01T00:00:00+00:00')
                ),
                (
                    'early_from_no_until',
                    parse('2018-03-01T00:00:00+00:00')
                ),
                (
                    'equal_from_no_until',
                    parse('2018-03-01T00:00:00+00:00')
                ),
                (
                    'late_from_no_until',
                    parse('2018-03-01T00:00:00+00:00')
                ),
                (
                    'no_from_late_until',
                    parse('2018-03-01T00:00:00+00:00')
                ),
            ]),
        )

    def test_screens_after_latest_datetime(self):
        datetime = parse('2018-03-01T00:00:01+00:00')
        video = AnnotatedVideo(INSTITUTION_ID, ROOM_ID, CAMERA_ID, datetime)
        frame = next(video.__iter__())
        self.assertEqual(
            set([
                (screen.screen_id, screen.datetime)
                for screen in self.screen_detector(frame)
            ]),
            set([
                (
                    'no_from_no_until',
                    parse('2018-03-01T00:00:00+00:00')
                ),
                (
                    'early_from_no_until',
                    parse('2018-03-01T00:00:00+00:00')
                ),
                (
                    'equal_from_no_until',
                    parse('2018-03-01T00:00:00+00:00')
                ),
                (
                    'late_from_no_until',
                    parse('2018-03-01T00:00:00+00:00')
                ),
            ]),
        )


if __name__ == '__main__':
    unittest.main()
