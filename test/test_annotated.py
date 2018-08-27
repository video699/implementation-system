# -*- coding: utf-8 -*-

import unittest

from dateutil.parser import parse
from video699.system.screen_detector.annotated import HumanAnnotationScreenDetector


def _screen_ids_at(datetime_str):
    """Detects ids of screens in the room 123 of the example institution at a given date, and time.

    Parameters
    ----------
    datetime_str : str
        A specification of date, and time in the ISO 8601 format.

    Returns
    -------
    screen_ids : set of str
        A set of the ids of detected lit projection screens.
    """
    datetime = parse(datetime_str)
    frame = None  # TODO: Use a Frame object
    screen_detector = HumanAnnotationScreenDetector('example', '123', datetime)
    screens = screen_detector.detect(frame)
    screen_ids = {
        screen[0] for screen in screens  # TODO: Use a ScreenABC object
    }
    return screen_ids


class TestHumanAnnotationScreenDetector(unittest.TestCase):
    """Tests the HumanAnnotationScreenDetector class from system.screen_detectors.annotated module.

    We test the ability of the class to correctly read, and interpret the example XML dataset.

    """

    def test_before_earliest(self):
        """No screen should be detected before the earliest date, and time.

        """
        screen_ids = _screen_ids_at('2017-12-31T23:59:59+00:00')
        self.assertEqual(screen_ids, set())

    def test_at_earliest(self):
        screen_ids = _screen_ids_at('2018-01-01T00:00:00+00:00')
        self.assertEqual(screen_ids, {
            'no_from_no_until',
            'early_from_no_until',
            'equal_from_no_until',
            'no_from_early_until',
            'no_from_equal_until',
            'no_from_late_until'
        })

    def test_after_earliest(self):
        screen_ids = _screen_ids_at('2018-01-01T00:00:01+00:00')
        self.assertEqual(screen_ids, {
            'no_from_no_until',
            'early_from_no_until',
            'equal_from_no_until',
            'late_from_no_until',
            'no_from_early_until',
            'no_from_equal_until',
            'no_from_late_until'
        })

    def test_before_latest(self):
        screen_ids = _screen_ids_at('2018-02-28T23:59:59+00:00')
        self.assertEqual(screen_ids, {
            'no_from_no_until',
            'early_from_no_until',
            'equal_from_no_until',
            'late_from_no_until',
            'no_from_equal_until',
            'no_from_late_until',
        })

    def test_at_latest(self):
        screen_ids = _screen_ids_at('2018-03-01T00:00:00+00:00')
        self.assertEqual(screen_ids, {
            'no_from_no_until',
            'early_from_no_until',
            'equal_from_no_until',
            'late_from_no_until',
            'no_from_late_until',
        })

    def test_after_latest(self):
        screen_ids = _screen_ids_at('2018-03-01T00:00:01+00:00')
        self.assertEqual(screen_ids, {
            'no_from_no_until',
            'early_from_no_until',
            'equal_from_no_until',
            'late_from_no_until',
        })


if __name__ == '__main__':
    unittest.main()
