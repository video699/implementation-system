# -*- coding: utf-8 -*-

r"""This module implements a screen event detector that matches document page image data with
projection screen image data using the Spearman's rank correlation coefficient :math:`\rho` with a
sliding window of video frames. Related classes and methods are also implemented.

"""

from collections import deque
from itertools import chain
from logging import getLogger

import cv2 as cv
import numpy as np
from scipy.stats import spearmanr

from ..common import benjamini_hochberg
from ..configuration import get_configuration
from ..interface import EventDetectorABC
from .screen import (
    ScreenAppearedEvent,
    ScreenChangedContentEvent,
    ScreenMovedEvent,
    ScreenDisappearedEvent,
)
from ..quadrangle.rtree import RTreeDequeConvexQuadrangleTracker


LOGGER = getLogger(__name__)
CONFIGURATION = get_configuration()['SlidingSpearmanEventDetector']
WINDOW_SIZE = int(CONFIGURATION['window_size'])
SIGNIFICANCE_LEVEL = float(CONFIGURATION['significance_level'])
SUBSAMPLE_SIZE = int(CONFIGURATION['subsample_size'])


class SlidingSpearmanEventDetector(EventDetectorABC):
    r"""A screen event detector using Spearman's rank correlation coefficient with a sliding window.

    In each frame of a video, screens are detected using a supplied screen detector. The detected
    screens are compared with the screens in the previous video frame. The detected screens that
    cross no previous screens and *match* a document page (more on that later) are reported in a
    :class:`ScreenAppearedEvent` event.  The detected screens that cross at least one previous
    screen are considered to correspond to the previous screen with the largest intersection area; a
    change of coordinates is reported in a :class:`ScreenMovedEvent` event and a change of a
    matching page is reported in a :class:`ScreenChangedContentEvent` event. The previous screens
    that cross none of the detected screens or that no longer match a page are reported in a
    :class:`ScreenDisappearedEvent` event.

    A random sample :math:`X` is taken from the intensities of the image data in a screen.
    Intensities with a corresponding alpha channel (A) value of zero are disregarded. A temporal
    sliding window is used to increase the sample size. Analogously, a random sample :math:`Y` of
    the same size is taken from the image data in a document page. Spearman's correlation
    coefficient :math:`\rho` between :math:`X` and :math:`Y` is computed. A significance test is
    performed to see if :math:`\rho` is sufficiently extreme to refuse the null hypothesis
    :math:`h_0: \rho = 0`. The page with the most extreme significant value of :math:`\rho` is said
    to *match* the screen. If no page has a significant value of :math:`\rho`, then no page matches
    the screen.

    Notes
    -----
    For screens that are detected in the last video frame of the video, a
    :class:`ScreenDisappearedEvent` event is not produced.

    Parameters
    ----------
    video : VideoABC
        The video in which the events are detected.
    screen_detector : ScreenDetectorABC
        A screen detector that will be used to detect lit projection screens in video frames.
    documents : set of DocumentABC
        Documents whose pages are matched against detected lit projection screens.

    Attributes
    ----------
    video : VideoABC
        The video in which the events are detected.
    """

    def __init__(self, video, screen_detector, documents):
        self._video = video
        self._screen_detector = screen_detector
        self._pages = list(chain(*documents))

    @property
    def video(self):
        return self._video

    def __iter__(self):
        quadrangle_tracker = RTreeDequeConvexQuadrangleTracker(WINDOW_SIZE)
        screen_detector = self._screen_detector
        pages = self._pages
        num_screens = 0
        screen_ids = {}
        matched_pages = {}
        matched_quadrangles = matched_pages.keys()
        samples = {}

        for frame in self.video:
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
                    screen = detected_screens[quadrangle]
                    screen_id = screen_ids[moving_quadrangle]
                    previous_page = matched_pages[moving_quadrangle]
                    del screen_ids[moving_quadrangle]
                    del matched_pages[moving_quadrangle]
                    del samples[moving_quadrangle]
                    LOGGER.debug('{} disappeared matching {}'.format(screen, previous_page))
                    yield ScreenDisappearedEvent(screen, screen_id)
                else:
                    LOGGER.debug('{} disappeared with no matching page'.format(screen))

            for moving_quadrangle in sorted(chain(appeared_quadrangles, existing_quadrangles)):
                p_values = []
                correlations = []

                quadrangle_iter = reversed(moving_quadrangle)
                current_quadrangle = next(quadrangle_iter)
                if moving_quadrangle in existing_quadrangles:
                    previous_quadrangle = next(quadrangle_iter)
                screen = detected_screens[current_quadrangle]
                screen_intensity = cv.cvtColor(screen.image, cv.COLOR_RGBA2GRAY)
                screen_alpha = screen.image[:, :, 3]

                if moving_quadrangle in appeared_quadrangles:
                    samples[moving_quadrangle] = {}

                for page in pages:
                    if moving_quadrangle in appeared_quadrangles:
                        screen_sample = deque(maxlen=WINDOW_SIZE)
                        page_sample = deque(maxlen=WINDOW_SIZE)
                        samples[moving_quadrangle][page] = (screen_sample, page_sample)
                    else:
                        screen_sample, page_sample = samples[moving_quadrangle][page]

                    page_image = page.image(screen.width, screen.height)
                    page_intensity = cv.cvtColor(page_image, cv.COLOR_RGBA2GRAY)
                    page_alpha = page_image[:, :, 3]

                    nonzero_alpha = np.minimum(screen_alpha, page_alpha).nonzero()
                    num_nonzero_alpha = len(nonzero_alpha[0])
                    pixel_subsample = np.random.choice(num_nonzero_alpha, SUBSAMPLE_SIZE)

                    screen_pixels = screen_intensity[nonzero_alpha][pixel_subsample]
                    page_pixels = page_intensity[nonzero_alpha][pixel_subsample]
                    screen_sample.append(screen_pixels)
                    page_sample.append(page_pixels)

                    correlation, p_value = spearmanr(
                        np.concatenate(screen_sample),
                        np.concatenate(page_sample),
                        nan_policy='raise',
                    )
                    correlations.append(correlation)
                    p_values.append(p_value)

                q_values = benjamini_hochberg(p_values)
                q_value_map = dict(zip(pages, q_values))
                correlation, page = max(zip(correlations, pages))
                q_value = q_value_map[page]

                if q_value > SIGNIFICANCE_LEVEL or correlation < 0:
                    if moving_quadrangle in appeared_quadrangles:
                        LOGGER.debug('{} appeared with no matching page'.format(screen))
                    elif moving_quadrangle in existing_quadrangles:
                        if moving_quadrangle in matched_quadrangles:
                            screen_id = screen_ids[moving_quadrangle]
                            previous_page = matched_pages[moving_quadrangle]
                            LOGGER.debug('{} no longer matches {}'.format(screen, previous_page))
                            del screen_ids[moving_quadrangle]
                            del matched_pages[moving_quadrangle]
                            yield ScreenDisappearedEvent(screen, screen_id)
                else:
                    if moving_quadrangle in appeared_quadrangles:
                        LOGGER.debug('{} appeared and matches {}'.format(screen, page))
                        screen_id = 'screen-{}'.format(num_screens)
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
                            screen_id = 'screen-{}'.format(num_screens + 1)
                            num_screens += 1
                            screen_ids[moving_quadrangle] = screen_id
                            yield ScreenAppearedEvent(screen, screen_id, page)
                    matched_pages[moving_quadrangle] = page
