# -*- coding: utf-8 -*-

"""This module implements a page of a document represented by a NumPy matrix containing image data.

"""

from functools import lru_cache

from ..interface import PageABC
from ..configuration import get_configuration

import cv2 as cv


CONFIGURATION = get_configuration()['ImagePage']
LRU_CACHE_MAXSIZE = CONFIGURATION.getint('lru_cache_maxsize')
RESCALING_INTERPOLATION = cv.__dict__[CONFIGURATION['rescaling_interpolation']]


class ImagePage(PageABC):
    """A page of a document represented by a NumPy matrix containing image data.

    Parameters
    ----------
    document : DocumentABC
        The document containing the page.
    number : int
        The page number, i.e. the position of the page in the document. Page indexing is one-based,
        i.e. the first page has number 1.
    image : ndarray
        The image data of the page represented as an OpenCV CV_8UC3 BGR matrix.

    Attributes
    ----------
    document : DocumentABC
        The document containing the page.
    number : int
        The page number, i.e. the position of the page in the document. Page indexing is one-based,
        i.e. the first page has number 1.
    """

    def __init__(self, document, number, image):
        self._document = document
        self._number = number
        self._image = image

    @property
    def document(self):
        return self._document

    @property
    def number(self):
        return self._number

    @lru_cache(maxsize=LRU_CACHE_MAXSIZE, typed=False)
    def image(self, width, height):
        image_rescaled = cv.resize(self._image, (width, height), RESCALING_INTERPOLATION)
        return image_rescaled
