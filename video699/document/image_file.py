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
    image : array_like
        The image data of the page as an OpenCV CV_8UC3 RGBA matrix, where the alpha channel (A)
        denotes the weight of a pixel. Fully transparent pixels, i.e. pixels with zero alpha, SHOULD
        be completely disregarded in subsequent computation. Any margins added to the image data,
        e.g. by keeping the aspect ratio of the page, MUST be fully transparent.
    """

    def __init__(self, document, number, image_pathname):
        self._document = document
        self._number = number
        self._hash = hash((self.number, self.document))
        self._image_pathname = image_pathname

    @property
    def document(self):
        return self._document

    @property
    def number(self):
        return self._number

    @lru_cache(maxsize=LRU_CACHE_MAXSIZE, typed=False)
    def render(self, width=None, height=None):
        bgr_image = cv.imread(self._image_pathname)
        rgba_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGBA)
        original_height, original_width, _ = rgba_image.shape
        rescaled_width, rescaled_height, top_margin, bottom_margin, left_margin, right_margin = \
            rescale_and_keep_aspect_ratio(original_width, original_height, width, height)
        rescale_interpolation = cv.__dict__[CONFIGURATION['rescale_interpolation']]
        rgba_image_rescaled = cv.resize(
            rgba_image,
            (rescaled_width, rescaled_height),
            rescale_interpolation,
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

    def __hash__(self):
        return self._hash


class ImageFileDocument(DocumentABC):
    """A document that consists of pages represented by NumPy matrices containing image data.

    .. _RFC3987: https://tools.ietf.org/html/rfc3987

    Parameters
    ----------
    image_pathnames : iterable of str
        The pathnames of the image files containing the individual pages in the document.
    title : str or None, optional
        The title of a document. ``None`` when unspecified.
    author : str or None, optional
        The author of a document. ``None`` when unspecified.

    Attributes
    ----------
    title : str or None
        The title of a document.
    author : str or None
        The author of a document.
    uri : string
        An IRI, as defined in RFC3987_, that uniquely indentifies the document over the entire
        lifetime of a program.

    Raises
    ------
    ValueError
        If no pathnames to image files containing document pages were provided.
    """

    _num_documents = 0

    def __init__(self, image_pathnames, title=None, author=None):
        self._title = title
        self._author = author

        self._uri = 'https://github.com/video699/implementation-system/blob/master/video699/' \
            'document/image_file.py#ImageFileDocument:{}'.format(self._num_documents + 1)
        self._num_documents += 1
        self._hash = hash(self._uri)

        self._pages = [
            ImageFileDocumentPage(self, page_number + 1, image_pathname)
            for page_number, image_pathname in enumerate(image_pathnames)
        ]
        if not self._pages:
            raise ValueError('No pathnames to image files containing document pages were provided')

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

    def __hash__(self):
        return self._hash
