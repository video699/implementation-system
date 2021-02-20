# -*- coding: utf-8 -*-

"""This module implements reading a sample of a video from a dataset with human annotations, and
related classes.

"""

from itertools import chain
import os

from video699.event.screen import (
    ScreenEventDetectorABC,
    ScreenAppearedEvent,
    ScreenChangedContentEvent,
    ScreenDisappearedEvent,
)


VIDEO_FILENAME = os.path.join(os.path.dirname(__file__), 'IA067-D2-20191112.mp4')
DOCUMENT_FILENAME = os.path.join(os.path.dirname(__file__), 'Blockchain-FI-MUNI.pdf')
ANNOTATIONS = {
    346: 1,
    1077: 2,
    1476: 3,
    1623: 4,
    2063: 5,
    2642: 6,
    3543: 7,
    4202: 8,
    4414: 9,
    4828: 10,
    5203: 11,
    5629: 12,
    6136: 13,
    6692: 14,
    7378: 15,
    8013: 16,
    9144: 17,
    10796: 18,
    11651: 19,
    11870: 20,
    12146: 21,
    12783: 22,
    13605: 23,
    14120: 24,
    14530: 25,
    14923: 26,
    15365: 27,
    15659: 28,
    15904: 29,
    16904: 30,
    18106: 31,
    18607: 32,
    19077: 33,
    19845: 34,
    20827: 35,
    21603: 36,
    22246: 37,
    22799: 38,
    23074: 39,
    23339: 40,
    23576: 41,
    23859: 42,
    24021: 43,
    24525: 44,
    25177: 45,
    25851: 46,
    27114: 47,
    27885: 48,
    34762: None,
    41686: 48,
    42005: 49,
    42657: 50,
    43460: 51,
    44285: 52,
    44916: 53,
    46540: 54,
    48310: 55,
    48789: 54,
    49148: 55,
    49553: 54,
    50002: 55,
    50334: 54,
    50395: 55,
    50409: 56,
    50477: 57,
    50572: 58,
    51308: 59,
    53715: 60,
    56038: 61,
    56730: 62,
    57659: 63,
    58526: 62,
    58548: 63,
    59328: 64,
    61884: 65,
    65925: None,
}


def evaluate_event_detector(event_detector):
    """Processes a video using a screen event detector and counts successful trials.

    A video file is processed using a screen event detector. When an annotated video frame is
    encountered, a trial takes place.  A trial is successful if and only if:

    1. there is an annotated page and the event detector has produced at least one
       and at most two screens, all with the correct page, or
    2. there is no annotated page and the event detector has produced no screens.

    Parameters
    ----------
    event_detector : ScreenEventDetectorABC
        The screen event detector.

    Returns
    -------
    num_successes : int
        The number of successful trials.
    num_trials : int
        The number of trials.

    """

    assert isinstance(event_detector, ScreenEventDetectorABC)

    remaining_annotated_frame_numbers = sorted(ANNOTATIONS, reverse=True)
    num_successes = 0
    num_trials = len(ANNOTATIONS)

    detected_page_dict = dict()
    for event in chain(event_detector, (None,)):
        # The None event processes all the remaining annotated frames at the end of a video
        while remaining_annotated_frame_numbers:
            frame_number = remaining_annotated_frame_numbers[-1]
            page_number = ANNOTATIONS[frame_number]
            if event is not None and frame_number >= event.frame.number:
                break
            remaining_annotated_frame_numbers.pop()

            if page_number is None:
                if not detected_page_dict:
                    num_successes += 1
            else:
                detected_page_numbers = set(page.number for page in detected_page_dict.values())
                if detected_page_numbers == set([page_number]):
                    num_successes += 1

        if isinstance(event, (ScreenAppearedEvent, ScreenChangedContentEvent)):
            detected_page_dict[event.screen_id] = event.page
        elif isinstance(event, ScreenDisappearedEvent):
            del detected_page_dict[event.screen_id]

    return (num_successes, num_trials)
