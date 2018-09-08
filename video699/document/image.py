# -*- coding: utf-8 -*-

"""This module implements a page of a document represented by a NumPy matrix containing image data.

"""

from functools import lru_cache

from ..interface import DocumentABC, PageABC
from ..configuration import get_configuration

import cv2 as cv


CONFIGURATION = get_configuration()['ImagePage']
LRU_CACHE_MAXSIZE = CONFIGURATION.getint('lru_cache_maxsize')
RESCALING_INTERPOLATION = cv.__dict__[CONFIGURATION['rescaling_interpolation']]


class ImageDocumentPage(PageABC):
    """A document page represented by a NumPy matrix containing image data.

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


class ImageDocument(DocumentABC):
    """A document that consists of pages represented by NumPy matrices containing image data.

    Parameters
    ----------
    page_images : iterable of array_like
        The image data of the individual pages represented as OpenCV CV_8UC3 BGR matrices.
    title : str or None, optional
        The title of a document. `None` when unspecified.
    author : str or None, optional
        The author of a document. `None` when unspecified.

    Attributes
    ----------
    title : str or None
        The title of a document.
    author : str or None
        The author of a document.
    """

    def __init__(self, page_images, title=None, author=None):
        self._title = title
        self._author = author
        self._pages = [
            ImageDocumentPage(self, page_number + 1, page_image)
            for page_number, page_image in enumerate(page_images)
        ]

    @property
    def title(self):
        pass

    @property
    def author(self):
        pass

    def __iter__(self):
        return iter(self._pages)
