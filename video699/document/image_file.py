# -*- coding: utf-8 -*-

"""This module implements a document represented by image files containing the individual pages.

"""

from functools import lru_cache

from ..interface import DocumentABC, PageABC
from ..configuration import get_configuration

import cv2 as cv


CONFIGURATION = get_configuration()['ImageFileDocumentPage']
LRU_CACHE_MAXSIZE = CONFIGURATION.getint('lru_cache_maxsize')
RESCALING_INTERPOLATION = cv.__dict__[CONFIGURATION['rescaling_interpolation']]


class ImageFileDocumentPage(PageABC):
    """A document page represented by a NumPy matrix containing image data.

    Parameters
    ----------
    document : DocumentABC
        The document containing the page.
    number : int
        The page number, i.e. the position of the page in the document. Page indexing is one-based,
        i.e. the first page has number 1.
    image_pathname : str
        The pathname of the image files containing the document page.

    Attributes
    ----------
    document : DocumentABC
        The document containing the page.
    number : int
        The page number, i.e. the position of the page in the document. Page indexing is one-based,
        i.e. the first page has number 1.
    """

    def __init__(self, document, number, image_pathname):
        self._document = document
        self._number = number
        self._image_pathname = image_pathname

    @property
    def document(self):
        return self._document

    @property
    def number(self):
        return self._number

    @lru_cache(maxsize=LRU_CACHE_MAXSIZE, typed=False)
    def image(self, width, height):
        image = cv.imread(self._image_pathname)
        image_rescaled = cv.resize(image, (width, height), RESCALING_INTERPOLATION)
        return image_rescaled


class ImageFileDocument(DocumentABC):
    """A document that consists of pages represented by NumPy matrices containing image data.

    Parameters
    ----------
    image_pathnames : iterable of str
        The pathnames of the image files containing the individual pages in the document.
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

    def __init__(self, image_pathnames, title=None, author=None):
        self._title = title
        self._author = author
        self._pages = [
            ImageFileDocumentPage(self, page_number + 1, image_pathname)
            for page_number, image_pathname in enumerate(image_pathnames)
        ]

    @property
    def title(self):
        pass

    @property
    def author(self):
        pass

    def __iter__(self):
        return iter(self._pages)
