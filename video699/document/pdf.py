# -*- coding: utf-8 -*-

"""This module implements reading a document from a PDF document file.

"""

import cv2 as cv
import fitz
import numpy as np

from ..interface import DocumentABC, PageABC


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
    number : int
        The page number, i.e. the position of the page in the document. Frame indexing is one-based,
        i.e. the first frame has number 1.
    """

    def __init__(self, document, page):
        self._document = document
        self._page = page
        pixmap = page.getPixmap()
        # Subtract 1, so that the dimensions of getPixmap(zoom_matrix) in image(self, width, height)
        # are never less than width x height due to subpixel errors.
        self._default_width = max(1, pixmap.width - 1)
        self._default_height = max(1, pixmap.height - 1)

    @property
    def document(self):
        return self._document

    @property
    def number(self):
        return self._page.number + 1

    def image(self, width, height):
        zoom_x = width / self._default_width
        zoom_y = height / self._default_height
        zoom_matrix = fitz.Matrix(zoom_x, zoom_y)
        pixmap = self._page.getPixmap(zoom_matrix, alpha=False)
        rgb_image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape((pixmap.h, pixmap.w, 3))
        bgr_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2RGB)
        bgr_image_downscaled = cv.resize(bgr_image, (width, height), cv.INTER_AREA)
        return bgr_image_downscaled


class PDFDocument(DocumentABC):
    """A PDF document read from a PDF document file.

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

    Raises
    ------
    ValueError
        If the pathname does not specify a PDF document file.
    """

    def __init__(self, pathname):
        self._document = fitz.open(pathname)
        if not self._document.isPDF:
            raise ValueError('The pathname "{}" does not specify a PDF document'.format(pathname))
        self._title = self._document.metadata['title']
        self._author = self._document.metadata['author']
        self._pages = [PDFDocumentPage(self, page) for page in self._document]

    @property
    def title(self):
        return self._title

    @property
    def author(self):
        return self._author

    def __iter__(self):
        return self._pages.__iter__()
