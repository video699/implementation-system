# -*- coding: utf-8 -*-

import os
import unittest

import cv2 as cv

from video699.document.pdf import PDFDocument


DOCUMENT_PATHNAME = os.path.join(
    os.path.dirname(__file__),
    'test_pdf',
    'sample_pdf_document.pdf',
)
DOCUMENT_TITLE = 'Example title'
DOCUMENT_AUTHOR = 'Example author'
PAGE_IMAGE_WIDTH = 2550
PAGE_IMAGE_HEIGHT = 3300


class TestPDFDocument(unittest.TestCase):
    """Tests the ability of the PDFDocument class to read a PDF document file.

    """

    def setUp(self):
        self.document = PDFDocument(DOCUMENT_PATHNAME)

    def test_document_properties(self):
        self.assertEqual(DOCUMENT_TITLE, self.document.title)
        self.assertEqual(DOCUMENT_AUTHOR, self.document.author)

    def test_reads_two_pages(self):
        page_iterator = iter(self.document)
        next(page_iterator)
        next(page_iterator)
        with self.assertRaises(StopIteration):
            next(page_iterator)


class TestPDFDocumentPage(unittest.TestCase):
    """Tests the ability of the PDFDocumentPage class to render pages and produce page numbers.

    """

    def setUp(self):
        self.document = PDFDocument(DOCUMENT_PATHNAME)
        page_iterator = iter(self.document)
        self.first_page = next(page_iterator)
        self.second_page = next(page_iterator)
        self.first_page_image = self.first_page.image(PAGE_IMAGE_WIDTH, PAGE_IMAGE_HEIGHT)
        self.second_page_image = self.second_page.image(PAGE_IMAGE_WIDTH, PAGE_IMAGE_HEIGHT)

    def test_document(self):
        self.assertEqual(self.document, self.first_page.document)
        self.assertEqual(self.document, self.second_page.document)

    def test_number(self):
        self.assertEqual(1, self.first_page.number)
        self.assertEqual(2, self.second_page.number)

    def test_first_page_image(self):
        height, width, _ = self.first_page_image.shape
        self.assertEqual(PAGE_IMAGE_WIDTH, width)
        self.assertEqual(PAGE_IMAGE_HEIGHT, height)

        blue, green, red = cv.split(self.first_page_image)

        self.assertEqual(255, blue[0, 0])
        self.assertEqual(255, green[0, 0])
        self.assertEqual(255, red[0, 0])

        self.assertEqual(0, blue[675, 900])
        self.assertEqual(0, green[675, 900])
        self.assertEqual(255, red[675, 900])

        self.assertEqual(255, blue[2200, 900])
        self.assertEqual(255, green[2200, 900])
        self.assertEqual(255, red[2200, 900])

        self.assertEqual(0, blue[2200, 1200])
        self.assertEqual(255, green[2200, 1200])
        self.assertEqual(0, red[2200, 1200])

    def test_second_page_image(self):
        height, width, _ = self.second_page_image.shape
        self.assertEqual(PAGE_IMAGE_WIDTH, width)
        self.assertEqual(PAGE_IMAGE_HEIGHT, height)

        blue, green, red = cv.split(self.second_page_image)

        self.assertEqual(255, blue[0, 0])
        self.assertEqual(255, green[0, 0])
        self.assertEqual(255, red[0, 0])

        self.assertEqual(255, blue[1250, 900])
        self.assertEqual(255, green[1250, 900])
        self.assertEqual(255, red[1250, 900])

        self.assertEqual(255, blue[1250, 1200])
        self.assertEqual(0, green[1250, 1200])
        self.assertEqual(0, red[1250, 1200])


if __name__ == '__main__':
    unittest.main()
