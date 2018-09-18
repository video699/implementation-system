# -*- coding: utf-8 -*-

"""This module implements a document represented by image files containing the individual pages.

"""

from functools import lru_cache

from ..interface import DocumentABC, PageABC
from ..common import COLOR_RGBA_TRANSPARENT, rescale_and_keep_aspect_ratio
from ..configuration import get_configuration

import cv2 as cv


CONFIGURATION = get_configuration()['ImageFileDocumentPage']
LRU_CACHE_MAXSIZE = CONFIGURATION.getint('lru_cache_maxsize')
RESCALE_INTERPOLATION = cv.__dict__[CONFIGURATION['rescale_interpolation']]


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
        bgr_image = cv.imread(self._image_pathname)
        rgba_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGBA)
        original_height, original_width, _ = rgba_image.shape
        rescaled_width, rescaled_height, top_margin, bottom_margin, left_margin, right_margin = \
            rescale_and_keep_aspect_ratio(original_width, original_height, width, height)
        rgba_image_rescaled = cv.resize(
            rgba_image,
            (
                rescaled_width,
                rescaled_height,
            ),
            RESCALE_INTERPOLATION
        )
        rgba_image_rescaled_with_margins = cv.copyMakeBorder(
            rgba_image_rescaled,
            top_margin,
            bottom_margin,
            left_margin,
            right_margin,
            borderType=cv.BORDER_CONSTANT,
            value=COLOR_RGBA_TRANSPARENT,
        )
        return rgba_image_rescaled_with_margins


class ImageFileDocument(DocumentABC):
    """A document that consists of pages represented by NumPy matrices containing image data.

    .. _RFC3987: https://tools.ietf.org/html/rfc3987

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
    uri : string
        An IRI, as defined in RFC3987_, that uniquely indentifies the document over the entire
        lifetime of a program.
    """

    _num_documents = 0

    def __init__(self, image_pathnames, title=None, author=None):
        self._title = title
        self._author = author
        self._pages = [
            ImageFileDocumentPage(self, page_number + 1, image_pathname)
            for page_number, image_pathname in enumerate(image_pathnames)
        ]
        self._uri = 'https://github.com/video699/implementation-system/blob/master/video699/' \
            'document/image_file.py#ImageFileDocument:{}'.format(ImageFileDocument._num_documents)
        ImageFileDocument._num_documents += 1

    @property
    def title(self):
        pass

    @property
    def author(self):
        pass

    @property
    def uri(self):
        return self._uri

    def __iter__(self):
        return iter(self._pages)
