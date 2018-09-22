# -*- coding: utf-8 -*-

r"""This module implements a screen event detector that matches document page image data with
projection screen image data using the Spearman's rank correlation coefficient :math:`\rho` with a
sliding window of video frames. Related classes are also implemented.

"""

from collections import deque
from itertools import chain

from ..configuration import get_configuration
from ..interface import EventDetectorABC
from ..quadrangle.rtree import RTreeConvexQuadrangleIndex
# from .screen import ScreenAppearedEvent, ScreenChangedContentEvent, ScreenMovedEvent, \
#     ScreenDisappearedEvent


CONFIGURATION = get_configuration()['SlidingSpearmanEventDetector']
WINDOW_SIZE = int(CONFIGURATION['window_size'])
# ALPHA = float(CONFIGURATION['alpha'])


class _TrackedScreen(object):
    """The movement of a uniquely identified lit projection screen in a sliding time window.

    Parameters
    ----------
    screen_id : str
        A screen identifier unique among the screen identifiers produced by a screen tracker.
    previous_screen : ScreenABC
        The screen detected in the previous video frame.

    Attributes
    ----------
    screen_id : str
        A screen identifier unique among the screen identifiers produced by a screen tracker.
    screens : deque of ScreenABC
        The movement of a lit projection screen in a sliding time window.
    previous_screen : ScreenABC
        The screen detected in the previous video frame.
    """

    def __init__(self, screen_id, previous_screen):
        self.screen_id = screen_id
        self.screens = deque((previous_screen,), maxlen=WINDOW_SIZE)

    @property
    def previous_screen(self):
        return self.screens[0]


class _ScreenTracker(object):
    """A tracker that records the movement of lit projection screens in a sliding time window.

    Attributes
    ----------
    tracked_screens : a set-like object of _TrackedScreen
        The tracked lit projection screens.
    """

    def __init__(self):
        self._screens = {}
        self._quadrangle_index = RTreeConvexQuadrangleIndex()

    def screens(self):
        return self._screens.values()

    def update(self, screens):
        """Records lit projection screens that were detected in a video frame.

        The detected lit projection screens are compared with the screens in the previous video
        frame. The detected screens that cross no previous screens are added to the tracker. The
        detected screens that cross at least one previous screen are linked to the previous screen
        with the largest intersection area. The previous screens that cross none of the detected
        screens are removed from the tracker.

        Parameters
        ----------
        screens : iterable of ScreenABC
            The lit projection screens that were detected in a video frame.

        Returns
        -------
        appeared : set of _TrackedScreen
            The detected screens that cross no previous screens.
        existed : set of _TrackedScreen
            The detected screens that cross at least one previous screen.
        disappeared : set of _TrackedScreen
            The previous screens that cross none of the detected screens.

        """
        # TODO
        pass


class SlidingSpearmanEventDetector(EventDetectorABC):
    r"""A screen event detector using Spearman's rank correlation coefficient with a sliding window.

    In each frame of a video, screens are detected using a supplied screen detector. The detected
    screens are compared with the screens in the previous video frame. The detected screens that
    cross no previous screens and *match* a document page (more on that later) are reported in a
    `ScreenAppearedEvent` event.  The detected screens that cross at least one previous screen are
    considered to correspond to the previous screen with the largest intersection area; a change of
    coordinates is reported in a `ScreenMovedEvent` event and a change of a matching page is
    reported in a `ScreenChangedContentEvent` event. The previous screens that cross none of the
    detected screens or that no longer match a page are reported in a `ScreenDisappearedEvent`
    event.

    Individual color components of image data pixels in a screen are taken to be a simple random
    sample :math:`X` (although the independence assumption is clearly violated). Color components
    with a corresponding alpha value of zero are disregarded. A temporal sliding window is used to
    increase the sample size. A document page produces a random sample :math:`Y` of the same size.
    Spearman's correlation coefficient :math:`\rho` between :math:`X` and :math:`Y` is computed. A
    significance test is performed to see if :math:`\rho` is sufficiently extreme to refuse the null
    hypothesis :math:`h_0: \rho = 0`. The page with the most extreme significant value of
    :math:`\rho` is said to *match* the screen. If no page has a significant value of :math:`\rho`,
    then no page matches the screen.

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
        self._pages = list(chain(documents))

    @property
    def video(self):
        return self._video

    def __iter__(self):
        # TODO
        pass
