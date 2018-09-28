# -*- coding: utf-8 -*-

r"""This module implements a screen event detector that matches document page image data with
projection screen image data using the Pearson's rank correlation coefficient :math:`r` with a
rolling window of video frames. Related classes and methods are also implemented.

"""

from collections import deque
from itertools import chain
from logging import getLogger

import cv2 as cv
import numpy as np
from scipy.stats import pearsonr

from ..common import benjamini_hochberg
from ..configuration import get_configuration
from ..interface import EventDetectorABC, PageDetectorABC
from .screen import ScreenEventDetector
from ..quadrangle.rtree import RTreeDequeConvexQuadrangleTracker


LOGGER = getLogger(__name__)
CONFIGURATION = get_configuration()['RollingPearsonEventDetector']
WINDOW_SIZE = int(CONFIGURATION['window_size'])
SIGNIFICANCE_LEVEL = float(CONFIGURATION['significance_level'])
SUBSAMPLE_SIZE = int(CONFIGURATION['subsample_size'])


class RollingPearsonPageDetector(PageDetectorABC):
    """A page detector using rolling Pearson's correlation coefficient.

    A random sample :math:`X` is taken from the intensities of the image data in a screen.
    Intensities with a corresponding alpha channel (A) value of zero are disregarded. A temporal
    rolling window is used to increase the sample size, i.e. the screens SHOULD originate from
    consecutive video frames. Analogously to :math:`X`, a random sample :math:`Y` of the same size
    is taken from the image data in a document page. Pearson's correlation coefficient :math:`r`
    between :math:`X` and :math:`Y` is computed. A significance test with the assumption that
    :math:`(X,Y)` is bivariate normal is performed to see if :math:`r` is sufficiently extreme to
    refuse the null hypothesis :math:`h_0: r = 0`. The page with the most extreme significant and
    positive value of :math:`r` is said to *match* the screen. If no page has a significant value of
    :math:`r`, then no page matches the screen.

    Parameters
    ----------
    documents : set of DocumentABC
        Documents whose pages are matched against detected lit projection screens.
    window_size : int
        The maximum number of previous video frames that contribute to the random samples from
        :math:`X`, and :math:`Y`.
    """

    def __init__(self, documents, window_size):
        self._pages = list(chain(*documents))
        self._window_size = window_size
        self._samples = {}

    def detect(self, appeared_screens, existing_screens, disappeared_screens):
        pages = self._pages
        window_size = self._window_size
        samples = self._samples
        detected_pages = {}

        for _, moving_quadrangle in disappeared_screens:
            del samples[moving_quadrangle]

        for screen, moving_quadrangle in chain(appeared_screens, existing_screens):
            p_values = []
            correlations = []

            if moving_quadrangle not in samples:
                samples[moving_quadrangle] = {}
            moving_quadrangle_samples = samples[moving_quadrangle]

            screen_intensity = cv.cvtColor(screen.image, cv.COLOR_RGBA2GRAY)
            screen_alpha = screen.image[:, :, 3]

            for page in pages:
                if page not in moving_quadrangle_samples:
                    screen_sample = deque(maxlen=window_size)
                    page_sample = deque(maxlen=window_size)
                    moving_quadrangle_samples[page] = (screen_sample, page_sample)
                else:
                    screen_sample, page_sample = moving_quadrangle_samples[page]

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

                correlation, p_value = pearsonr(
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
            detected_page = None
            if q_value < SIGNIFICANCE_LEVEL and correlation > 0:
                detected_page = page
            detected_pages[screen] = detected_page

        return detected_pages


class RTreeDequeRollingPearsonEventDetector(EventDetectorABC):
    r"""A screen event detector that wraps :class:`ScreenEventDetector` and serves as a facade.

    A :class:`ScreenEventDetector` is instantiated with the
    :class:`RTreeDequeConvexQuadrangleTracker` convex quadrangle tracker and the
    :class:`RollingPearsonPageDetector` page detector. The window size for the convex quadrangle
    tracker and for the page detector is taken from the configuration.

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
        quadrangle_tracker = RTreeDequeConvexQuadrangleTracker(WINDOW_SIZE)
        page_detector = RollingPearsonPageDetector(documents, WINDOW_SIZE)
        self._event_detector = ScreenEventDetector(
            video,
            quadrangle_tracker,
            screen_detector,
            page_detector,
        )

    @property
    def video(self):
        return self._event_detector.video

    def __iter__(self):
        return iter(self._event_detector)
