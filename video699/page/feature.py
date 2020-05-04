# -*- coding: utf-8 -*-

"""This module implements a screen event detector that matches local feature descriptors extracted
from document page image data with local feature descriptors extracted from projection screen image
data using nearest neighbor search. Related classes and functions are also implemented.

"""

from bisect import bisect
from collections import deque
from functools import lru_cache
from itertools import chain
from logging import getLogger

from annoy import AnnoyIndex
import cv2 as cv

from ..configuration import get_configuration
from ..interface import PageDetectorABC


LOGGER = getLogger(__name__)
CONFIGURATION = get_configuration()['LocalFeatureKNNPageDetector']
LRU_CACHE_MAXSIZE = CONFIGURATION.getint('lru_cache_maxsize')
NUM_FEATURE_DIMENSIONS = 32


@lru_cache()
def _get_feature_detector(num_features):
    """Returns an ORB feature detector.

    Parameters
    ----------
    num_features : int
        The number of ORB features detected by the detector.

    Returns
    -------
    feature_detector : cv.ORB
        The ORB feature detector.
    """

    feature_detector = cv.ORB_create(num_features)
    return feature_detector


@lru_cache(maxsize=LRU_CACHE_MAXSIZE, typed=False)
def _extract_features(image):
    """Extracts local features from an image.

    Parameters
    ----------
    image : ImageABC
        An image from which we will extract local features.

    Returns
    -------
    descriptors : array_like or None
        Local feature descriptors in the format returned by ``cv.Feature2D.compute()``. The
        descriptors are ``None`` if and only if no keypoints were extracted.
    """

    image_intensity = cv.cvtColor(image.image, cv.COLOR_RGBA2GRAY)
    image_alpha = image.image[:, :, 3]
    keypoints = []
    num_features = CONFIGURATION.getint('num_features')
    feature_detector = _get_feature_detector(num_features)
    for keypoint in feature_detector.detect(image_intensity):
        x, y = keypoint.pt
        if image_alpha[int(y), int(x)] > 0:
            keypoints.append(keypoint)
    _, descriptors = feature_detector.compute(image_intensity, keypoints)
    return descriptors


class RollingMean(object):
    """An efficient implementation of a rolling mean.

    Parameters
    ----------
    window_size : int or None, optional
        The size of the rolling window, i.e. the maximum number of previous time frames for which
        :math:`r` is computed. If ``None`` or unspecified, then the number of time frames is
        unbounded.
    """

    def __init__(self, window_size=None):
        if window_size is not None and window_size < 1:
            raise ValueError('The window size must not be less than one')
        self._window_size = window_size

        self._sample = deque(maxlen=window_size)

        self.clear()

    def clear(self):
        """Clears the sample.

        """

        self._sample.clear()
        self._mean_sum = 0

    def next(self, observation):
        r"""Adds an observation to the sample, moves to the next time frame, computes rolling mean.

        Parameters
        ----------
        observation : scalar
            An observation that will be added to the sample.

        Returns
        -------
        mean : scalar
            A pointwise estimate of the rolling mean.

        """

        window_size = self._window_size

        sample = self._sample
        mean_sum = self._mean_sum

        if len(sample) == window_size:
            popped_observation = sample[0]

            mean_sum -= popped_observation

        mean_sum += observation

        sample.append(observation)
        self._mean_sum = mean_sum

        mean = mean_sum / len(sample)

        return mean


class LocalFeatureKNNPageDetector(PageDetectorABC):
    """A page detector using approximate nearest neighbor search of local image features.

    Local features are extracted from the image data of the provided document pages and placed
    inside a vector database. Local features are then extracted from the image data in a screen
    and the nearest neighbors of every feature are retrieved from the vector database. The document
    page with the most nearest neighbors averaged across a small time window is detected as the page
    shown in the screen. If only few local features are extracted from the screen image data or if
    only few nearest neighbors are retrieved, then no page is detected in the screen.

    This technique was developed by [ParedesEtAl01]_ and shown to give state-of-the-art results for
    content-based information retrieval (CBIR) by [DeselaersEtAl07]_.

    .. [ParedesEtAl01] Paredes, Roberto, et al. "Local representations and a direct voting scheme
       for face recognition." In *Workshop on Pattern Recognition in Information Systems*. 2001.
    .. [DeselaersEtAl07] Deselaers, Thomas, Daniel Keysers, and Hermann Ney. "Features for image
       retrieval: an experimental comparison." *Information retrieval* 11.2 (2008): 77-107.

    Parameters
    ----------
    documents : set of DocumentABC
        The provided document pages.
    window_size : int
        The maximum number of previous video frames that participate in the voting.
    """

    def __init__(self, documents, window_size):
        annoy_n_trees = CONFIGURATION.getint('annoy_n_trees')
        distance_metric = CONFIGURATION['distance_metric']
        min_page_vote_percentage = CONFIGURATION.getfloat('min_page_vote_percentage')
        num_features = CONFIGURATION.getint('num_features')
        num_nearest_features = CONFIGURATION.getint('num_nearest_features')
        min_num_screen_features = int(
            CONFIGURATION.getfloat('min_feature_percentage') * num_features
        )
        min_num_page_features = int(
            min_page_vote_percentage * num_nearest_features * min_num_screen_features
        )

        LOGGER.debug('Building an ANNOY index with {} trees'.format(annoy_n_trees))
        annoy_index = AnnoyIndex(NUM_FEATURE_DIMENSIONS, metric=distance_metric)
        pages = []
        page_indices = []
        base_index = 0
        for page in chain(*documents):
            page_descriptors = _extract_features(page)
            if page_descriptors is None or len(page_descriptors) < min_num_page_features:
                LOGGER.warn('Skipping {}, since we extracted no or few local features'.format(page))
                continue
            pages.append(page)
            page_indices.append(base_index)
            num_page_descriptors, _ = page_descriptors.shape
            for index, page_descriptor in enumerate(page_descriptors):
                annoy_index.add_item(base_index + index, page_descriptor)
            base_index += num_page_descriptors
        annoy_index.build(annoy_n_trees)

        self._window_size = window_size
        self._annoy_index = annoy_index
        self._pages = pages
        self._page_indices = page_indices
        self._rolling_means = {}
        self._previous_frame = None

    def detect(self, frame, appeared_screens, existing_screens, disappeared_screens):
        annoy_search_k = CONFIGURATION.getint('annoy_search_k')
        num_features = CONFIGURATION.getint('num_features')
        num_nearest_features = CONFIGURATION.getint('num_nearest_features')
        min_num_screen_features = int(
            CONFIGURATION.getfloat('min_feature_percentage') * num_features
        )
        min_page_vote_percentage = CONFIGURATION.getfloat('min_page_vote_percentage')

        window_size = self._window_size
        annoy_index = self._annoy_index
        pages = self._pages
        page_indices = self._page_indices
        rolling_means = self._rolling_means
        previous_frame = self._previous_frame

        if previous_frame is not None and frame.number != previous_frame.number + 1:
            for rolling_mean in rolling_means:
                rolling_mean.clean()
        previous_frame = frame
        self._previous_frame = previous_frame

        detected_pages = {}

        for _, moving_quadrangle in disappeared_screens:
            del rolling_means[moving_quadrangle]

        for screen, moving_quadrangle in chain(appeared_screens, existing_screens):
            screen_descriptors = _extract_features(screen)

            detected_pages[screen] = None

            if screen_descriptors is None:
                continue

            num_screen_descriptors, _ = screen_descriptors.shape

            if num_screen_descriptors < min_num_screen_features:
                continue

            page_votes = {}
            for screen_descriptor in screen_descriptors:
                descriptor_indices = annoy_index.get_nns_by_vector(
                    screen_descriptor,
                    num_nearest_features,
                    search_k=annoy_search_k,
                )
                for descriptor_index in descriptor_indices:
                    page_index = bisect(page_indices, descriptor_index) - 1
                    page = pages[page_index]
                    if page not in page_votes:
                        page_votes[page] = 0
                    page_votes[page] += 1

            if moving_quadrangle not in rolling_means:
                rolling_means[moving_quadrangle] = {}

            page_mean_votes = {}
            for page, num_page_votes in page_votes.items():
                if page not in rolling_means[moving_quadrangle]:
                    rolling_means[moving_quadrangle][page] = RollingMean(window_size)
                rolling_mean = rolling_means[moving_quadrangle][page]
                mean_num_page_votes = rolling_mean.next(num_page_votes)
                page_mean_votes[page] = mean_num_page_votes

            page, mean_num_page_votes = max(page_mean_votes.items(), key=lambda x: (x[1], x[0]))

            min_mean_num_page_votes = int(
                min_page_vote_percentage * num_nearest_features * len(screen_descriptors)
            )
            if mean_num_page_votes >= min_mean_num_page_votes:
                detected_pages[screen] = page

        return detected_pages
