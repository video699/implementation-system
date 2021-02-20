# -*- coding: utf-8 -*-

"""This module implements reading a document from a PDF document file.

"""

from functools import lru_cache
from logging import getLogger
from pathlib import Path

import cv2 as cv
import fitz
import numpy as np

from video699.common import COLOR_RGBA_TRANSPARENT, rescale_and_keep_aspect_ratio
from video699.configuration import get_configuration
from video699.interface import DocumentABC, PageABC


LOGGER = getLogger(__name__)
CONFIGURATION = get_configuration()['PDFDocumentPage']
LRU_CACHE_MAXSIZE = CONFIGURATION.getint('lru_cache_maxsize')


class PDFDocumentPage(PageABC):
    """A page of a PDF document read from a PDF document file.

    Parameters
    ----------
    document : DocumentABC
        The document containing the page.
    page : fitz.Page
        The internal representation of the page by the PyMuPDF library.

    Attributes
    ----------
    document : DocumentABC
        The document containing the page.
    image : array_like
        The image data of the page as an OpenCV CV_8UC3 RGBA matrix, where the alpha channel (A)
        denotes the weight of a pixel. Fully transparent pixels, i.e. pixels with zero alpha, SHOULD
        be completely disregarded in subsequent computation. Any margins added to the image data,
        e.g. by keeping the aspect ratio of the page, MUST be fully transparent.
    number : int
        The page number, i.e. the position of the page in the document. Page indexing is one-based,
        i.e. the first page has number 1.
    """

    def __init__(self, document, page):
        self._document = document
        self._page = page
        self._hash = hash((self.number, self.document))
        pixmap = page.getPixmap()
        self._default_width = pixmap.width
        self._default_height = pixmap.height

    @property
    def document(self):
        return self._document

    @property
    def number(self):
        return self._page.number + 1

    @property
    def image(self):
        rgba_image = self.render()
        return rgba_image

    @lru_cache(maxsize=LRU_CACHE_MAXSIZE, typed=False)
    def render(self, width=None, height=None):
        rescaled_width, rescaled_height, top_margin, bottom_margin, left_margin, right_margin = \
            rescale_and_keep_aspect_ratio(self._default_width, self._default_height, width, height)
        # Subtract 1, so that the dimensions of getPixmap(zoom_matrix) are never less than
        # rescaled_width x rescaled_height due to subpixel errors.
        zoom_x = rescaled_width / max(1, self._default_width - 1)
        zoom_y = rescaled_height / max(1, self._default_height - 1)
        zoom_matrix = fitz.Matrix(zoom_x, zoom_y)
        pixmap = self._page.getPixmap(zoom_matrix, alpha=False)
        rgb_image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape((pixmap.h, pixmap.w, 3))
        rgba_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2RGBA)
        downscale_interpolation = cv.__dict__[CONFIGURATION['downscale_interpolation']]
        rgba_image_downscaled = cv.resize(
            rgba_image,
            (rescaled_width, rescaled_height),
            downscale_interpolation,
        )
        rgba_image_downscaled_with_margins = cv.copyMakeBorder(
            rgba_image_downscaled,
            top_margin,
            bottom_margin,
            left_margin,
            right_margin,
            borderType=cv.BORDER_CONSTANT,
            value=COLOR_RGBA_TRANSPARENT,
        )
        return rgba_image_downscaled_with_margins

    def __hash__(self):
        return self._hash


class PDFDocument(DocumentABC):
    """A PDF document read from a PDF document file.

    .. _RFC3987: https://tools.ietf.org/html/rfc3987

    Note
    ----
    A document file is opened as soon as the class is instantiated, and closed only after the
    finalization of the object.

    Parameters
    ----------
    pathname : str
        The pathname of a PDF document file.

    Attributes
    ----------
    title : str
        The title of a document.
    author : str
        The author of a document.
    pathname : str
        The pathname of a PDF document file.
    uri : string
        An IRI, as defined in RFC3987_, that uniquely indentifies the document over the entire
        lifetime of a program.

    Raises
    ------
    ValueError
        If the pathname does not specify a PDF document file or if the PDF document contains no
        pages.
    """

    def __init__(self, pathname):
        self.pathname = pathname
        self._uri = Path(pathname).resolve().as_uri()
        self._hash = hash(self._uri)

        self._document = fitz.open(pathname)
        if not self._document.isPDF:
            raise ValueError('The pathname "{}" does not specify a PDF document'.format(pathname))

        self._title = self._document.metadata['title']
        self._author = self._document.metadata['author']

        LOGGER.debug('Loading PDF document {}'.format(pathname))
        self._pages = [PDFDocumentPage(self, page) for page in self._document]
        if not self._pages:
            raise ValueError('PDF document at "{}" contains no pages'.format(pathname))

    @property
    def title(self):
        return self._title

    @property
    def author(self):
        return self._author

    @property
    def uri(self):
        return self._uri

    def __iter__(self):
        return iter(self._pages)

    def __hash__(self):
        return self._hash
