# -*- coding: utf-8 -*-

r"""This module implements a screen event detector that matches image hashes extracted from document
page image data with image hashes extracted from projection screen image data. Related classes and
functions are also implemented.

"""


from itertools import chain
from logging import getLogger

from annoy import AnnoyIndex
import cv2 as cv
import imagehash
from PIL import Image

from video699.configuration import get_configuration
from video699.interface import PageDetectorABC
from video699.event.screen import ScreenEventDetector, ScreenEventDetectorABC
from video699.quadrangle.rtree import RTreeDequeConvexQuadrangleTracker


LOGGER = getLogger(__name__)
CONFIGURATION = get_configuration()['ImageHashPageDetector']


def _hash_image(image):
    r"""Produces an image hash for an image.

    Parameters
    ----------
    image : ImageABC
        An image.

    Returns
    -------
    image_hash : array_like
        A hash of the image.
    """

    image_cv = cv.cvtColor(image.image, cv.COLOR_BGRA2RGBA)
    image_pil = Image.fromarray(image_cv, 'RGBA')
    hash_function = imagehash.__dict__[CONFIGURATION['hash_function']]
    return hash_function(image_pil).hash.flatten()


class ImageHashPageDetector(PageDetectorABC):
    r"""A page detector using approximate nearest neighbor search of image hashes.

    The following image hashes are used to represent images:
    - average hash and perception hash [Krawetz11]_,
    - difference hash [Krawetz13]_, and
    - wavelet hash [Petrov16]_.

    .. [Krawetz11]. Krawetz, Neal. "Looks Like It." *The Hacker Factor Blog*. 2011.
       `URL <http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html>`_
    .. [Krawetz13]. Krawetz, Neal. "Kind of Like That." *The Hacker Factor Blog*. 2013.
       `URL <http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html>`_
    .. [Petrov16]. Petrov, Dmitri. "Wavelet image hash in Python." *FullStackML*. 2016.
       `URL <https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5>`_

    Parameters
    ----------
    documents : set of DocumentABC
        The provided document pages.
    """

    def __init__(self, documents):
        annoy_n_trees = CONFIGURATION.getint('annoy_n_trees')
        annoy_distance_metric = CONFIGURATION['distance_metric']
        LOGGER.debug('Building an ANNOY index with {} trees'.format(annoy_n_trees))
        annoy_index = AnnoyIndex(64, metric=annoy_distance_metric)
        pages = dict()
        for page_index, page in enumerate(chain(*documents)):
            page_hash = _hash_image(page)
            annoy_index.add_item(page_index, page_hash)
            pages[page_index] = page
        annoy_index.build(annoy_n_trees)

        self._annoy_index = annoy_index
        self._pages = pages

    def detect(self, frame, appeared_screens, existing_screens, disappeared_screens):
        annoy_search_k = CONFIGURATION.getint('annoy_search_k')
        num_nearest_pages = CONFIGURATION.getint('num_nearest_pages')
        max_distance = CONFIGURATION.getint('max_distance')

        annoy_index = self._annoy_index
        pages = self._pages

        detected_pages = {}
        screens = set(screen for screen, _ in chain(appeared_screens, existing_screens))
        for screen in screens:
            screen_hash = _hash_image(screen)
            page_indices, page_distances = annoy_index.get_nns_by_vector(
                screen_hash,
                num_nearest_pages,
                search_k=annoy_search_k,
                include_distances=True,
            )
            closest_matching_page = None
            for page_index, page_distance in zip(page_indices, page_distances):
                if page_distance < max_distance:
                    closest_matching_page = pages[page_index]
                    break
            detected_pages[screen] = closest_matching_page

        return detected_pages


class RTreeDequeImageHashEventDetector(ScreenEventDetectorABC):
    r"""A screen event detector that wraps :class:`ImageHashPageDetector` and serves as a facade.

    A :class:`ScreenEventDetector` is instantiated with the
    :class:`RTreeDequeConvexQuadrangleTracker` convex quadrangle tracker and the
    :class:`ImageHashPageDetector` page detector.

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
        page_detector = ImageHashPageDetector(documents)
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
