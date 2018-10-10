# -*- coding: utf-8 -*-

r"""This module implements a screen event detector that matches document page image data with
projection screen image data using the Pearson's rank correlation coefficient :math:`r` with a
rolling window of video frames. Related classes and functions are also implemented.

"""

from collections import deque
from functools import lru_cache
from itertools import chain
from math import sqrt

import cv2 as cv
import numpy as np
from scipy.special import stdtr

from ..common import COLOR_RGBA_TRANSPARENT, benjamini_hochberg
from ..configuration import get_configuration
from ..interface import PageDetectorABC
from .screen import ScreenEventDetector, ScreenEventDetectorABC
from ..quadrangle.rtree import RTreeDequeConvexQuadrangleTracker


CONFIGURATION = get_configuration()['RollingPearsonPageDetector']
LRU_CACHE_MAXSIZE = CONFIGURATION.getint('lru_cache_maxsize')
WINDOW_SIZE = CONFIGURATION.getint('window_size')
CORRELATION_THRESHOLD = CONFIGURATION.getfloat('correlation_threshold')
SIGNIFICANCE_LEVEL = CONFIGURATION.getfloat('significance_level')
SAMPLE_SIZE = CONFIGURATION.getint('sample_size')
FEATURE_DETECTOR = cv.ORB_create(CONFIGURATION.getint('num_features'))
DESCRIPTOR_MATCHER = cv.DescriptorMatcher_create(CONFIGURATION['descriptor_matcher_type'])
GOOD_MATCH_PERCENTAGE = CONFIGURATION.getfloat('good_match_percentage')
FIND_HOMOGRAPHY_METHOD = cv.__dict__[CONFIGURATION['find_homography_method']]
RESCALE_INTERPOLATION = cv.__dict__[CONFIGURATION['rescale_interpolation']]


@lru_cache(maxsize=LRU_CACHE_MAXSIZE, typed=False)
def _extract_features(image):
    """Extracts local features from an image.

    Parameters
    ----------
    image : ImageABC
        An image from which we will extract local features.

    Returns
    -------
    keypoints : list of KeyPoint
        Local features in the format returned by ``cv.Feature2D.compute()``.
    descriptors : array_like or None
        Local feature descriptors in the format returned by ``cv.Feature2D.compute()``. The
        descriptors are ``None`` if and only if no keypoints were extracted.
    """

    image_intensity = cv.cvtColor(image.image, cv.COLOR_RGBA2GRAY)
    image_alpha = image.image[:, :, 3]
    keypoints = []
    for keypoint in FEATURE_DETECTOR.detect(image_intensity):
        x, y = keypoint.pt
        if image_alpha[int(y), int(x)] > 0:
            keypoints.append(keypoint)
    _, descriptors = FEATURE_DETECTOR.compute(image_intensity, keypoints)
    if descriptors is not None:
        # Flann-based descriptor matcher requires the descriptors to be 32-bit floats.
        descriptors_float32 = np.float32(descriptors)
    else:
        descriptors_float32 = None
    return (keypoints, descriptors_float32)


def _find_homography(image, template):
    """Finds a homography that aligns a template with an image.

    Parameters
    ----------
    image : ImageABC
        An image.
    template : ImageABC
        A template that we will align with the image.

    Returns
    -------
    transform_matrix : 3x3 ndarray of scalar
        The homography that aligns the template with the image.
    """

    image_keypoints, image_descriptors = _extract_features(image)
    template_keypoints, template_descriptors = _extract_features(template)

    transform_matrix = None

    if len(image_keypoints) >= 4 and len(template_keypoints) >= 4:
        matches = DESCRIPTOR_MATCHER.match(template_descriptors, image_descriptors)
        matches.sort(key=lambda match: match.distance)
        num_good_matches = int(len(matches) * GOOD_MATCH_PERCENTAGE)

        if num_good_matches >= 4:
            good_matches = matches[:num_good_matches]
            template_points = np.zeros((len(good_matches), 2))
            image_points = np.zeros((len(good_matches), 2))
            for index, match in enumerate(good_matches):
                template_points[index, :] = template_keypoints[match.queryIdx].pt
                image_points[index, :] = image_keypoints[match.trainIdx].pt
            transform_matrix, _ = cv.findHomography(
                template_points,
                image_points,
                FIND_HOMOGRAPHY_METHOD,
            )

    if transform_matrix is None:
        transform_matrix = np.identity(3)

    return transform_matrix


class RollingPearsonR(object):
    """An efficient implementation of rolling weighted Pearson's correlation coefficient :math:`r`.

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

        self._sample_x = deque(maxlen=window_size)
        self._sample_y = deque(maxlen=window_size)
        self._weights = deque(maxlen=window_size)

        self.clear()

    def clear(self):
        """Clears the sample.

        """

        self._sample_x.clear()
        self._mean_x_sum = 0
        self._mean_x2_sum = 0

        self._sample_y.clear()
        self._mean_y_sum = 0
        self._mean_y2_sum = 0

        self._weights.clear()
        self._weights_sum = 0

        self._mean_xy_sum = 0

    def next(self, observations_x, observations_y, observation_weights, update=True):
        r"""Adds provided observations to sample, moves to the next time frame, computes :math:`r`.

        Parameters
        ----------
        observations_x : array_like
            The provided observations of the first random variable.
        observations_y : array_like
            The provided observations of the second random variable.
        observation_weights : array_like
            Non-negative weights of the observations of the bivariate random vector.
        update : bool, optional
            If ``True``, then :math:`r` is computed, but the provided observations are not actually
            added to the bivariate random sample, and we do not actually move to the next time
            frame. If `False`` or unspecified, then the observation are added to the sample and we
            move to the next time frame.

        Returns
        -------
        correlation_coefficient : scalar
            A pointwise estimate of the rolling weighted Pearson's correlation coefficient
            :math:`r`.
        p_value : scalar
            The probability of obtaining an estimate of :math:`r` that is at least as extreme as the
            current estimate under the null hypothesis that :math:`r = 0`.

        Raises
        ------
        ValueError
            When ``observations_x``, ``observations_y``, and ``observation_weights`` have different
            lengths.
        """

        if not len(observations_x) == len(observations_y) == len(observation_weights):
            raise ValueError('Arrays containing bivariate observations must have the same length')

        window_size = self._window_size

        sample_x = self._sample_x
        mean_x_sum = self._mean_x_sum
        mean_x2_sum = self._mean_x2_sum

        sample_y = self._sample_y
        mean_y_sum = self._mean_y_sum
        mean_y2_sum = self._mean_y2_sum

        weights = self._weights
        weights_sum = self._weights_sum

        mean_xy_sum = self._mean_xy_sum

        if len(sample_x) == window_size:
            popped_observations_x = np.asfarray(sample_x[0])
            popped_observations_y = np.asfarray(sample_y[0])
            popped_weights = np.asfarray(weights[0])

            mean_x_sum -= popped_observations_x.dot(popped_weights)
            mean_x2_sum -= (popped_observations_x**2).dot(popped_weights)

            mean_y_sum -= popped_observations_y.dot(popped_weights)
            mean_y2_sum -= (popped_observations_y**2).dot(popped_weights)

            weights_sum -= popped_weights.sum()

            mean_xy_sum -= (popped_weights * popped_observations_x).dot(popped_observations_y)

        observations_x = np.asfarray(observations_x)
        observations_y = np.asfarray(observations_y)
        observation_weights = np.asfarray(observation_weights)

        mean_x_sum += observations_x.dot(observation_weights)
        mean_x2_sum += (observations_x**2).dot(observation_weights)

        mean_y_sum += observations_y.dot(observation_weights)
        mean_y2_sum += (observations_y**2).dot(observation_weights)

        weights_sum += observation_weights.sum()

        mean_xy_sum += (observation_weights * observations_x).dot(observations_y)

        degrees_of_freedom = weights_sum - 2

        if degrees_of_freedom > 0:
            mean_x = mean_x_sum / weights_sum
            mean_x2 = mean_x2_sum / weights_sum
            var_x = np.clip(mean_x2 - mean_x**2, 0, None)
            std_x = sqrt(var_x)

            mean_y = mean_y_sum / weights_sum
            mean_y2 = mean_y2_sum / weights_sum
            var_y = np.clip(mean_y2 - mean_y**2, 0, None)
            std_y = sqrt(var_y)

            mean_xy = mean_xy_sum / weights_sum
            cov_xy = mean_xy - (mean_x * mean_y)

            if std_x > 0 and std_y > 0:
                pearsonr = np.clip(cov_xy / (std_x * std_y), 0, 1)

                if pearsonr < 1:
                    t_star = pearsonr * sqrt(degrees_of_freedom) / sqrt(1 - pearsonr**2)
                    p_left_tail = stdtr(degrees_of_freedom, -abs(t_star))
                    p_value = np.clip(2 * p_left_tail, 0, 1)
                else:
                    p_value = 0.0
            else:
                pearsonr = 0.0
                p_value = 0.0
        else:
            pearsonr = 0.0
            p_value = 0.0

        if update:
            sample_x.append(observations_x)
            self._mean_x_sum = mean_x_sum
            self._mean_x2_sum = mean_x2_sum

            sample_y.append(observations_y)
            self._mean_y_sum = mean_y_sum
            self._mean_y2_sum = mean_y2_sum

            weights.append(observation_weights)
            self._weights_sum = weights_sum

            self._mean_xy_sum = mean_xy_sum

        return pearsonr, p_value


class RollingPearsonPageDetector(PageDetectorABC):
    """A page detector using rolling Pearson's correlation coefficient.

    A random sample :math:`X` is taken from the intensities of the image data in a projection
    screen.  The alpha channel (A) in the original RGBA image data weighs the intensities. A
    small time window is used to increase the sample size, i.e. the screens SHOULD originate from
    consecutive video frames. Analogously to :math:`X`, a random sample :math:`Y` of the same size
    is taken from the image data in a document page. Weighted Pearson's correlation coefficient
    :math:`r` between :math:`X` and :math:`Y` is computed. A significance test with the assumption
    that :math:`(X,Y)` is bivariate normal is performed to see if :math:`r` is sufficiently extreme
    to refuse the null hypothesis :math:`h_0: r = 0`. The page with the most extreme significant and
    positive value of :math:`r` is detected as the page shown in the screen. If no page has a
    significant value of :math:`r`, then no page is detected in the screen.

    Parameters
    ----------
    documents : set of DocumentABC
        Documents whose pages are matched against detected lit projection screens.
    window_size : int
        The maximum number of previous video frames that contribute to the random samples from
        :math:`X`, and :math:`Y`.
    """

    def __init__(self, documents, window_size):
        pages = list(chain(*documents))
        self._pages = pages
        self._window_size = window_size
        self._rolling_pearsons = {}
        self._correlation_coefficients = np.empty(len(pages))
        self._p_values = np.empty(len(pages))
        self._previous_frame = None

    def detect(self, frame, appeared_screens, existing_screens, disappeared_screens):
        pages = self._pages
        window_size = self._window_size
        rolling_pearsons = self._rolling_pearsons
        correlation_coefficients = self._correlation_coefficients
        p_values = self._p_values
        previous_frame = self._previous_frame

        if previous_frame is not None and frame.number != previous_frame.number + 1:
            for rolling_pearson in rolling_pearsons:
                rolling_pearson.clean()
        previous_frame = frame
        self._previous_frame = previous_frame

        detected_pages = {}

        for _, moving_quadrangle in disappeared_screens:
            del rolling_pearsons[moving_quadrangle]

        for screen, moving_quadrangle in chain(appeared_screens, existing_screens):
            page_pixel_samples = []

            if moving_quadrangle not in rolling_pearsons:
                rolling_pearsons[moving_quadrangle] = {}

            pixel_sample = (
                np.random.random_integers(0, screen.height - 1, SAMPLE_SIZE),
                np.random.random_integers(0, screen.width - 1, SAMPLE_SIZE),
            )

            screen_intensity = cv.cvtColor(screen.image, cv.COLOR_RGBA2GRAY)
            screen_pixels = screen_intensity[pixel_sample]
            screen_alpha = screen.image[:, :, 3][pixel_sample]

            for page_index, page in enumerate(pages):
                if page not in rolling_pearsons[moving_quadrangle]:
                    rolling_pearsons[moving_quadrangle][page] = RollingPearsonR(window_size)
                rolling_pearson = rolling_pearsons[moving_quadrangle][page]

                transform_matrix = _find_homography(screen, page)
                page_image = cv.warpPerspective(
                    page.image,
                    transform_matrix,
                    (screen.width, screen.height),
                    borderMode=cv.BORDER_CONSTANT,
                    borderValue=COLOR_RGBA_TRANSPARENT,
                    flags=RESCALE_INTERPOLATION,
                )
                page_intensity = cv.cvtColor(page_image, cv.COLOR_RGBA2GRAY)
                page_pixels = page_intensity[pixel_sample]
                page_alpha = page_image[:, :, 3][pixel_sample]

                pixel_weights = np.minimum(screen_alpha, page_alpha) / 255

                correlation_coefficient, p_value = rolling_pearson.next(
                    screen_pixels,
                    page_pixels,
                    pixel_weights,
                    update=False,
                )
                correlation_coefficients[page_index] = correlation_coefficient
                p_values[page_index] = p_value
                page_pixel_samples.append((page_pixels, pixel_weights))

            page_index = np.argmax(correlation_coefficients)
            correlation_coefficient = correlation_coefficients[page_index]
            q_value = benjamini_hochberg(p_values)[page_index]
            page = pages[page_index]
            page_pixels, pixel_weights = page_pixel_samples[page_index]

            rolling_pearson = rolling_pearsons[moving_quadrangle][page]
            rolling_pearson.next(screen_pixels, page_pixels, pixel_weights)

            if q_value <= SIGNIFICANCE_LEVEL and correlation_coefficient >= CORRELATION_THRESHOLD:
                detected_page = page
            else:
                detected_page = None
            detected_pages[screen] = detected_page

        return detected_pages


class RTreeDequeRollingPearsonEventDetector(ScreenEventDetectorABC):
    r"""A screen event detector that wraps :class:`ScreenEventDetector` and serves as a facade.

    A :class:`ScreenEventDetector` is instantiated with the
    :class:`RTreeDequeConvexQuadrangleTracker` convex quadrangle tracker and the
    :class:`RollingPearsonPageDetector` page detector. The window size for the page detector is
    taken from the configuration.

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
        quadrangle_tracker = RTreeDequeConvexQuadrangleTracker(2)
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
